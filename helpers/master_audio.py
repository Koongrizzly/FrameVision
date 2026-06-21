"""
master_audio.py
----------------

Experimental realtime master audio engine for FrameVision.

Goals / Design:
- No temporary WAV renders on disk.
- Low-latency "live" playback using streaming decode + EQ in Python.
- 12-band graphic EQ (60 Hz .. 14 kHz) using cascaded biquad peak filters.
- Master volume + mute.
- Threaded decoder so UI stays responsive.
- Tiny hook surface for framevision_app.VideoPane.

Integration plan (minimal edits to framevision_app.py / VideoPane.__init__):
    from helpers.master_audio import MasterAudio
    self.master_audio = MasterAudio()
    self.experimental_audio_engine = True

In VideoPane open/play logic for AUDIO files ONLY, do:

    if getattr(self, "experimental_audio_engine", False):
        self.master_audio.load(path_to_file)
        self.master_audio.play()
        # (update labels like info_label/time_label exactly like you do today)
        return

Volume / EQ popup (volume_new.py) should *not* try to poke QMediaPlayer
for audio files when experimental_audio_engine is True. Instead call:

    pane.master_audio.set_volume(scalar_0_to_1)
    pane.master_audio.set_muted(bool)
    pane.master_audio.set_eq_band(index, gain_db)
    pane.master_audio.set_all_eq_bands(gain_list_db)

IMPORTANT:
- This module intentionally does NOT touch video playback or A/V sync.
  Video keeps using QMediaPlayer for now.
- Seeking / duration reporting / waveform preview are TODO for later.

Runtime deps:
- numpy
- ffmpeg available on PATH or in ./bin/ like the rest of FrameVision
- pyaudio (or PortAudio wrapper) for actual sound device output.
  If pyaudio is missing, we silently fall back to a "Null" sink so the
  app will not crash. You just won't hear anything.

Thread model:
- MasterAudio._worker_thread runs tight loop:
    * if paused -> wait
    * read small chunk from ffmpeg stdout
    * apply EQ + volume/mute
    * write to sound device
- All control methods (.play(), .pause(), .stop(), etc.) are thread-safe.

This file tries to stay self-contained and readable. It's not final-prod
DSP code. It's a first working drop-in that we can iterate on.
"""

from __future__ import annotations

import os
import math
import threading
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Utility: locate ffmpeg just like framevision_app.py does
# ---------------------------------------------------------------------------

def _ffmpeg_path() -> str:
    """
    Best-effort locator for ffmpeg.

    FrameVision currently ships ffmpeg binaries in ./presets/bin,
    NOT ./bin. We still fall back to ./bin and then system PATH.

    We keep this local so we don't import the giant framevision_app
    just to ask for the path.
    """
    root = Path('.').resolve()
    candidates = [
        root / "presets" / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"),
        root / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"),
        "ffmpeg",
    ]
    for cand in candidates:
        try:
            subprocess.check_output([str(cand), "-version"], stderr=subprocess.STDOUT)
            return str(cand)
        except Exception:
            continue
    return "ffmpeg"  # final fallback, hope it's on PATH


# ---------------------------------------------------------------------------
# Biquad peaking EQ filter
# ---------------------------------------------------------------------------

class _BiquadPeakingEQ:
    """
    Simple biquad "peaking" EQ stage, implemented in Direct Form II Transposed.

    y[n] = b0*x[n] + z1
    z1'  = b1*x[n] - a1*y[n] + z2
    z2'  = b2*x[n] - a2*y[n]

    Coeffs assume a0 == 1 (we normalize when designing).
    """

    __slots__ = ("b0","b1","b2","a1","a2","z1","z2")

    def __init__(self, b0=1.0,b1=0.0,b2=0.0,a1=0.0,a2=0.0):
        self.b0 = float(b0)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.z1 = 0.0
        self.z2 = 0.0

    def reset_state(self):
        self.z1 = 0.0
        self.z2 = 0.0

    def process_inplace(self, x: np.ndarray) -> None:
        """
        In-place filter on a 1-D float32 array.
        We keep Python loops here for clarity; perf can be tuned later.
        """
        # local vars for speed
        b0 = self.b0; b1 = self.b1; b2 = self.b2
        a1 = self.a1; a2 = self.a2
        z1 = self.z1; z2 = self.z2

        # x is float32 ndarray, we'll operate sample-by-sample
        # (Later optimization could use numba/Cython/etc.)
        for i in range(x.shape[0]):
            xi = float(x[i])
            yi = b0 * xi + z1
            z1_new = b1 * xi - a1 * yi + z2
            z2     = b2 * xi - a2 * yi
            x[i] = yi  # write back
            z1 = z1_new
        self.z1 = z1
        self.z2 = z2

    @staticmethod
    def design_peaking(fs: float, f0: float, gain_db: float, Q: float=1.1) -> "_BiquadPeakingEQ":
        """
        Return a new _BiquadPeakingEQ configured as a peaking EQ filter.

        fs       sample rate (Hz)
        f0       center frequency (Hz)
        gain_db  boost/cut in dB (-12..+12 typical)
        Q        quality factor. ~1.1 gives ~1-octave-ish width.
        """
        # Special case: ~0 dB == bypass (identity filter). Keeps math stable.
        if abs(gain_db) < 1e-6:
            return _BiquadPeakingEQ(1.0,0.0,0.0,0.0,0.0)

        A  = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * (f0 / fs)
        alpha = math.sin(w0) / (2.0 * Q)
        cosw0 = math.cos(w0)

        b0 = 1.0 + alpha*A
        b1 = -2.0 * cosw0
        b2 = 1.0 - alpha*A
        a0 = 1.0 + alpha/A
        a1 = -2.0 * cosw0
        a2 = 1.0 - alpha/A

        # normalize so a0 == 1
        b0 /= a0; b1 /= a0; b2 /= a0
        a1 /= a0; a2 /= a0
        return _BiquadPeakingEQ(b0,b1,b2,a1,a2)


# ---------------------------------------------------------------------------
# MasterAudio: thread-based streaming player with 12-band EQ
# ---------------------------------------------------------------------------

class MasterAudio:
    """
    Minimal public API we agreed on:

        load(path)
        play()
        pause()
        stop()
        set_volume(scalar_0_to_1)
        set_muted(bool)
        set_eq_band(band_index, gain_db)
        set_all_eq_bands(gain_list_db)

    "position", "duration", "seek" will come later.

    Implementation notes:
    - We decode the source file with ffmpeg to float32 stereo @ 48 kHz.
    - We feed small chunks (default ~1024 frames) to PortAudio (PyAudio).
    - We apply 12 peaking EQ bands (graphic EQ style) to both L and R.
    - All heavy work is in a background thread.
    """

    # Frequencies must match the UI sliders in volume_new.BANDS order.
    _EQ_BANDS_HZ = [
        60,   120,   230,   460,
        910,  1800,  2500,  3600,
        5000, 7000, 10000, 14000,
    ]

    def __init__(self, sample_rate:int=48000, channels:int=2, chunk_frames:int=1024):
        self.sample_rate   = int(sample_rate)
        self.channels      = int(channels)
        self.chunk_frames  = int(chunk_frames)

        # playback state
        self._path: Optional[Path] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag  = threading.Event()
        self._play_event = threading.Event()  # "is playing?"
        self._thread_lock = threading.Lock()  # protect thread create/teardown

        # runtime control params
        self._param_lock = threading.Lock()
        self._volume     = 1.0     # scalar 0..1
        self._muted      = False
        self._gains_db   = [0.0]*12  # per-band gain in dB

        # filter chains for L and R. Each is list[_BiquadPeakingEQ].
        self._filters_L: List[_BiquadPeakingEQ] = []
        self._filters_R: List[_BiquadPeakingEQ] = []
        self._rebuild_filters_locked()  # init

        # "engine" objects that live inside thread:
        # ffmpeg process, PyAudio stream, etc.
        # We don't create them here; worker creates & destroys.
        # We keep a weak-ish snapshot for debug/inspection only.
        self._debug_backend_ok = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | os.PathLike) -> None:
        """
        Prepare a new file for playback.
        Stops any existing playback thread, resets filters state,
        and remembers the new path. Does NOT auto-play.
        """
        p = Path(path)
        # Stop current thread / audio first to avoid device fights.
        self.stop()

        self._path = p

        # Reset filter states so we don't carry resonance from old song.
        with self._param_lock:
            for f in self._filters_L:
                f.reset_state()
            for f in self._filters_R:
                f.reset_state()

    def play(self) -> None:
        """
        If no worker thread is alive, spawn it.
        Then set play_event so audio actually flows.
        """
        with self._thread_lock:
            # Spawn if needed
            if (self._thread is None) or (not self._thread.is_alive()):
                self._stop_flag.clear()
                self._play_event.clear()  # start paused; we'll set below
                self._thread = threading.Thread(
                    target=self._worker_main,
                    name="MasterAudioThread",
                    daemon=True
                )
                self._thread.start()

        # now mark as "playing"
        self._play_event.set()

    def pause(self) -> None:
        """
        Pause playback but keep decoder thread + ffmpeg alive.
        Resume later with play().
        """
        self._play_event.clear()

    def stop(self) -> None:
        """
        Hard stop. Kill playback thread and release audio device.
        After stop(), play() will restart from the *beginning* of self._path.
        """
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                self._stop_flag.set()
                # Unblock thread if it's paused
                self._play_event.set()
                try:
                    self._thread.join(timeout=1.0)
                except Exception:
                    pass
            self._thread = None
        self._stop_flag.clear()
        self._play_event.clear()

    def set_volume(self, scalar_0_to_1: float) -> None:
        """
        Master volume in linear scalar (0.0 .. 1.0).
        """
        try:
            vol = float(scalar_0_to_1)
        except Exception:
            vol = 1.0
        if vol < 0.0:
            vol = 0.0
        if vol > 1.0:
            vol = 1.0
        with self._param_lock:
            self._volume = vol

    def set_muted(self, is_muted: bool) -> None:
        """
        Master mute. When True we just multiply by 0.0.
        """
        with self._param_lock:
            self._muted = bool(is_muted)

    def set_eq_band(self, band_index: int, gain_db: float) -> None:
        """
        Update a single EQ band gain (in dB), rebuild filter chain.
        band_index: 0..11 (60 Hz .. 14 kHz)
        gain_db:    -12.0 .. +12.0 (UI already clamps this)
        """
        if band_index < 0 or band_index >= len(self._gains_db):
            return
        try:
            g = float(gain_db)
        except Exception:
            g = 0.0
        if g < -24.0:
            g = -24.0
        if g >  24.0:
            g = 24.0
        with self._param_lock:
            self._gains_db[band_index] = g
            self._rebuild_filters_locked()

    def set_all_eq_bands(self, gain_list_db: List[float]) -> None:
        """
        Bulk update of all 12 bands.
        gain_list_db must be len==12.
        """
        if not isinstance(gain_list_db, (list,tuple)):
            return
        if len(gain_list_db) != len(self._gains_db):
            return
        # sanitize / clamp
        cleaned = []
        for g in gain_list_db:
            try:
                v = float(g)
            except Exception:
                v = 0.0
            if v < -24.0: v = -24.0
            if v >  24.0: v =  24.0
            cleaned.append(v)
        with self._param_lock:
            self._gains_db[:] = cleaned
            self._rebuild_filters_locked()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_filters_locked(self) -> None:
        """
        Recompute biquad chain for each channel based on current _gains_db.
        NOTE: Caller must hold self._param_lock.
        Each band is an independent peaking filter, then we cascade them.
        """
        fs = float(self.sample_rate)
        new_L = []
        new_R = []
        for idx, freq in enumerate(self._EQ_BANDS_HZ):
            gain_db = float(self._gains_db[idx])
            fL = _BiquadPeakingEQ.design_peaking(fs, float(freq), gain_db, Q=1.1)
            fR = _BiquadPeakingEQ.design_peaking(fs, float(freq), gain_db, Q=1.1)
            new_L.append(fL)
            new_R.append(fR)
        self._filters_L = new_L
        self._filters_R = new_R

    def _open_ffmpeg_proc(self, path: Path):
        """
        Launch ffmpeg to decode `path` to raw float32 stereo at sample_rate.
        Returns subprocess.Popen with stdout=PIPE.
        """
        cmd = [
            _ffmpeg_path(),
            "-v", "error",
            "-i", str(path),
            "-f", "f32le",
            "-ac", str(self.channels),
            "-acodec", "pcm_f32le",
            "-ar", str(self.sample_rate),
            "pipe:1",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            return proc
        except Exception as e:
            print("[MasterAudio] ffmpeg spawn failed:", e)
            return None

    def _open_audio_device(self):
        """
        Open the OS audio device using PyAudio in blocking-write mode.
        Returns (pyaudio_instance, stream) or (None, None) if failed.
        """
        try:
            import pyaudio
        except Exception:
            # Graceful "null sink"
            return (None, None)

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_frames,
            )
            return (pa, stream)
        except Exception as e:
            print("[MasterAudio] PyAudio open() failed:", e)
            try:
                pa.terminate()
            except Exception:
                pass
            return (None, None)

    def _close_audio_device(self, pa, stream) -> None:
        """
        Safe cleanup for PyAudio.
        """
        try:
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if pa is not None:
                pa.terminate()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    
    def _worker_main(self):
        """
        Main playback loop. Runs in self._thread.
        This function blocks until file finishes or stop() is called.

        NOTE ON PERFORMANCE / UI HANGS:
        If PyAudio isn't available (or fails to open), we used to decode
        the entire file in a tight loop as fast as ffmpeg could push data.
        That pegged the CPU/GIL and starved the Qt UI thread, making the
        whole app feel frozen.

        Now:
        - If we *don't* have a real audio device, we still spawn ffmpeg,
          but we throttle the loop to "realtime" using time.sleep() and
          we skip the heavy EQ math entirely. You won't hear audio
          (obviously, no device), but the UI stays responsive.
        - As soon as PyAudio is available in your build, playback will be
          realtime with live EQ.
        """
        # Snapshot the currently loaded path NOW.
        path = self._path
        if path is None:
            return

        proc = self._open_ffmpeg_proc(path)
        if proc is None or proc.stdout is None:
            return

        pa, stream = self._open_audio_device()
        have_device = not (pa is None or stream is None)
        if have_device:
            self._debug_backend_ok = True
        else:
            self._debug_backend_ok = False

        bytes_per_frame = self.channels * 4  # float32 stereo => 8 bytes/frame at 2ch
        chunk_bytes = self.chunk_frames * bytes_per_frame

        try:
            while (not self._stop_flag.is_set()):
                # --- Handle pause ---
                if not self._play_event.is_set():
                    # We're "paused". We don't want ffmpeg to race ahead,
                    # so we just block here until resumed or stopped.
                    for _ in range(20):
                        if self._stop_flag.is_set():
                            break
                        if self._play_event.is_set():
                            break
                        time.sleep(0.05)
                    continue
                if self._stop_flag.is_set():
                    break

                # --- Read raw float32 PCM from ffmpeg ---
                try:
                    raw = proc.stdout.read(chunk_bytes)
                except Exception:
                    raw = b""
                if not raw:
                    # EOF or error => we're done.
                    break

                # Convert to numpy for DSP
                # shape: (chunk_frames, channels)
                try:
                    buf = np.frombuffer(raw, dtype=np.float32)
                except Exception:
                    # decoding glitch -> bail out
                    break
                if buf.size == 0:
                    break
                # Make sure we have N x channels even for last partial chunk
                frames = buf.size // self.channels
                if frames <= 0:
                    break
                buf = buf[:frames*self.channels].reshape(frames, self.channels)

                # Fast path for "no actual audio device": drop audio but throttle so we don't
                # busy-spin and lock the UI.
                if not have_device:
                    # Sleep approximately the chunk duration to simulate realtime.
                    # No EQ / volume math so we don't waste CPU/GIL.
                    chunk_dur = frames / float(self.sample_rate or 48000)
                    if chunk_dur > 0:
                        time.sleep(chunk_dur)
                    continue

                # --- Apply EQ + volume ---
                # Copy so we can safely write in-place
                out = buf.copy()

                with self._param_lock:
                    # snapshot runtime params for this chunk
                    muted = self._muted
                    vol   = self._volume
                    filtersL = self._filters_L
                    filtersR = self._filters_R

                # channel 0 (Left)
                left = out[:,0]
                for f in filtersL:
                    f.process_inplace(left)

                # channel 1 (Right)
                if self.channels > 1:
                    right = out[:,1]
                    for f in filtersR:
                        f.process_inplace(right)

                # master volume / mute
                if muted:
                    out *= 0.0
                else:
                    out *= float(vol)

                # --- Write to audio device ---
                if stream is not None:
                    try:
                        # We'll send contiguous interleaved float32
                        stream.write(out.astype(np.float32, copy=False).tobytes(),
                                     exception_on_underflow=False)
                    except Exception as e:
                        print("[MasterAudio] stream.write failed:", e)
                        break

                # loop continues until stop_flag set, pause() called, or EOF
        finally:
            # cleanup
            try:
                if proc and (proc.poll() is None):
                    try:
                        proc.kill()
                    except Exception:
                        pass
            except Exception:
                pass

            self._close_audio_device(pa, stream)
            try:
                if proc and proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc and proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

            # Mark ourselves fully stopped
            self._stop_flag.clear()
            self._play_event.clear()
            # NOTE: we do NOT clear self._path. Calling play() again without
            # load() will simply restart from beginning of same file.

