# -*- coding: utf-8 -*-
"""Telegram remote-control bridge for FrameVision.

This first remote layer deliberately does not require the local LLM.  It uses
FrameVisionAssistantRouter for deterministic create-image/create-video wizards,
and provides lightweight queue/status/cancel/last-result commands.

The bridge uses Telegram Bot API long polling (outbound HTTPS only): no port
forwarding or inbound web server is required.
"""
from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import shutil
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from PySide6 import QtCore
except Exception:  # pragma: no cover
    QtCore = None  # type: ignore

_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
_VIDEO_EXTS = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}
_AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus'}


def _root_path(root: str | Path) -> Path:
    return Path(root).resolve()


def _jobs_dirs(root: Path) -> Dict[str, Path]:
    base = root / 'jobs'
    return {name: base / name for name in ('pending', 'running', 'done', 'failed')}


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(str(tmp), str(path))


def _all_job_json_files(root: Path, bucket: str) -> List[Path]:
    """Return JSON files in a queue bucket without assuming every JSON is a job.

    Some FrameVision helpers leave *.progress.json marker/state files in
    jobs/running, and completed job JSON can occasionally remain there too.
    Those must not be reported as active queue jobs.
    """
    p = _jobs_dirs(root).get(bucket)
    if not p or not p.exists():
        return []
    try:
        return sorted((x for x in p.glob('*.json') if x.is_file()),
                      key=lambda x: x.stat().st_mtime, reverse=True)
    except Exception:
        return []


def _looks_like_progress_marker(path: Path, data: Optional[Dict[str, Any]] = None) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    if '.progress' in name or stem.endswith('.progress') or '_progress' in stem:
        return True
    d = data if isinstance(data, dict) else _read_json(path)
    kind = ' '.join(str(d.get(k) or '') for k in ('job_type', 'type', 'engine', 'kind', 'status')).lower()
    # Do not classify a normal job as a marker merely because its status says
    # "in progress"; require explicit marker-ish type/name language.
    return any(token in kind for token in ('progress_marker', 'progress file', 'heartbeat marker'))


def _job_is_finished(data: Dict[str, Any]) -> bool:
    status = str(data.get('status') or data.get('state') or '').strip().lower()
    if bool(data.get('done') or data.get('completed') or data.get('finished')):
        return True
    return status in {
        'done', 'complete', 'completed', 'finished', 'success', 'succeeded',
        'failed', 'error', 'cancelled', 'canceled'
    } or status.startswith(('done:', 'completed:', 'finished:', 'success:', 'failed:', 'error:'))


def _job_files(root: Path, bucket: str) -> List[Path]:
    """Return actual queue jobs, filtering marker files and stale finished jobs."""
    out: List[Path] = []
    for path in _all_job_json_files(root, bucket):
        data = _read_json(path)
        if _looks_like_progress_marker(path, data):
            continue
        if bucket in {'running', 'pending'} and _job_is_finished(data):
            continue
        out.append(path)
    return out


def _ignored_running_files(root: Path) -> int:
    all_files = _all_job_json_files(root, 'running')
    active = {str(p) for p in _job_files(root, 'running')}
    return sum(1 for p in all_files if str(p) not in active)


def worker_status(root: Path) -> tuple[bool, str]:
    """Best-effort check using the heartbeat written by helpers/worker.py."""
    hb = root / 'logs' / 'worker_heartbeat.txt'
    try:
        if hb.exists():
            age = max(0.0, time.time() - hb.stat().st_mtime)
            if age <= 10.0:
                return True, f'Worker: running (heartbeat {age:.0f}s ago)'
            return False, f'Worker: not detected (last heartbeat {age:.0f}s ago)'
    except Exception:
        pass
    return False, 'Worker: not detected'

def queue_summary(root: Path) -> str:
    d = _jobs_dirs(root)
    counts = {k: len(_job_files(root, k)) for k in d}
    running = _job_files(root, 'running')
    pending = _job_files(root, 'pending')
    _worker_ok, _worker_text = worker_status(root)
    lines = [_worker_text, f"Queue: {counts['running']} running, {counts['pending']} pending, {counts['done']} done, {counts['failed']} failed."]
    ignored = _ignored_running_files(root)
    if ignored:
        lines.append(f"Ignored {ignored} stale/progress file{'s' if ignored != 1 else ''} in jobs/running.")
    for label, files in (('Running', running[:3]), ('Pending', pending[:5])):
        if files:
            lines.append(label + ':')
            for f in files:
                j = _read_json(f)
                jt = str(j.get('job_type') or j.get('type') or j.get('engine') or f.stem)
                title = str(j.get('title') or j.get('name') or '').strip()
                lines.append('• ' + (title or jt))
    return '\n'.join(lines)


def cancel_current_job(root: Path) -> str:
    running = _job_files(root, 'running')
    if running:
        p = running[0]
        data = _read_json(p)
        data['cancel_requested'] = True
        # Do NOT mark a still-running job as already cancelled here. The worker
        # owns the final cancelled/failed state after it actually observes the
        # request and stops at a safe checkpoint.
        data['status'] = 'Cancel requested from Telegram'
        try:
            _write_json_atomic(p, data)

            # worker.py natively watches <running-job>.json.cancel as well as the
            # cancel_requested JSON field. Write both so cancellation is reliable
            # even while another process is frequently rewriting progress JSON.
            cancel_marker = Path(str(p) + '.cancel')
            try:
                cancel_marker.write_text('cancel requested from Telegram\n', encoding='utf-8')
            except Exception:
                # JSON flag is still a valid fallback.
                pass

            jt = str(data.get('job_type') or data.get('type') or data.get('engine') or '').lower()
            if 'planner' in jt:
                # Also drop Planner's project-local safe-stop marker. This is
                # consumed by PipelineWorker at its normal between-image/clip
                # checkpoints, matching the Planner GUI's "finish current item,
                # then stop" behavior.
                out_dir = str(data.get('out_dir') or '').strip()
                if not out_dir:
                    for container_key in ('payload', 'job', 'params', 'data'):
                        nested = data.get(container_key)
                        if isinstance(nested, dict) and str(nested.get('out_dir') or '').strip():
                            out_dir = str(nested.get('out_dir')).strip()
                            break
                if out_dir:
                    try:
                        marker = Path(out_dir) / '_telegram_cancel_after_current.flag'
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.write_text('cancel requested from Telegram\n', encoding='utf-8')
                    except Exception:
                        pass
                return (
                    f"Planner cancel requested. The current clip/image may finish, "
                    f"then the Planner will stop. ({p.stem})"
                )
            return f"Cancel requested for running job: {p.stem}"
        except Exception as exc:
            return f"Could not request cancellation: {exc}"
    pending = _job_files(root, 'pending')
    if pending:
        p = pending[0]
        failed = _jobs_dirs(root)['failed']
        failed.mkdir(parents=True, exist_ok=True)
        data = _read_json(p)
        data['cancelled'] = True
        data['status'] = 'Cancelled from Telegram before start'
        try:
            _write_json_atomic(p, data)
            shutil.move(str(p), str(failed / p.name))
            return f"Cancelled pending job: {p.stem}"
        except Exception as exc:
            return f"Could not cancel pending job: {exc}"
    return 'There is no running or pending FrameVision job to cancel.'


def find_last_result(root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    roots = [root / 'output']
    exts = _IMAGE_EXTS | _VIDEO_EXTS | _AUDIO_EXTS
    for base in roots:
        if not base.exists():
            continue
        try:
            for p in base.rglob('*'):
                if p.is_file() and p.suffix.lower() in exts:
                    candidates.append(p)
        except Exception:
            continue
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception:
        return candidates[0]


class TelegramApi:
    def __init__(self, token: str, timeout: int = 35):
        self.token = str(token or '').strip()
        self.base = f'https://api.telegram.org/bot{self.token}'
        self.file_base = f'https://api.telegram.org/file/bot{self.token}'
        self.timeout = int(timeout)

    def _json_call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        data = urllib.parse.urlencode({k: str(v) for k, v in (params or {}).items() if v is not None}).encode('utf-8')
        req = urllib.request.Request(self.base + '/' + method, data=data)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            obj = json.loads(resp.read().decode('utf-8', errors='replace'))
        if not obj.get('ok'):
            raise RuntimeError(str(obj.get('description') or 'Telegram API call failed'))
        return obj.get('result')

    def get_me(self) -> Dict[str, Any]:
        return dict(self._json_call('getMe') or {})

    def get_updates(self, offset: int = 0, timeout: int = 25) -> List[Dict[str, Any]]:
        return list(self._json_call('getUpdates', {'offset': offset, 'timeout': timeout, 'allowed_updates': json.dumps(['message'])}) or [])

    def send_message(self, chat_id: str, text: str) -> None:
        clean = str(text or '').strip() or 'Done.'
        # Telegram limits a message to 4096 characters.
        while clean:
            chunk = clean[:3900]
            clean = clean[3900:]
            self._json_call('sendMessage', {'chat_id': chat_id, 'text': chunk})

    def get_file_path(self, file_id: str) -> str:
        info = dict(self._json_call('getFile', {'file_id': file_id}) or {})
        return str(info.get('file_path') or '')

    def download_file(self, file_path: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(self.file_base + '/' + file_path, timeout=60) as resp, destination.open('wb') as out:
            shutil.copyfileobj(resp, out)
        return destination

    def send_file(self, chat_id: str, path: Path, caption: str = '') -> None:
        path = Path(path)
        if not path.exists():
            return
        ext = path.suffix.lower()
        method = 'sendPhoto' if ext in _IMAGE_EXTS else 'sendVideo' if ext in _VIDEO_EXTS else 'sendAudio' if ext in _AUDIO_EXTS else 'sendDocument'
        field = {'sendPhoto': 'photo', 'sendVideo': 'video', 'sendAudio': 'audio'}.get(method, 'document')
        boundary = '----FrameVisionTelegram' + uuid.uuid4().hex
        parts: List[bytes] = []
        def add_field(name: str, value: str) -> None:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        add_field('chat_id', str(chat_id))
        if caption:
            add_field('caption', caption[:900])
        mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        header = (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{path.name}"\r\nContent-Type: {mime}\r\n\r\n').encode()
        parts.append(header + path.read_bytes() + b'\r\n')
        parts.append(f'--{boundary}--\r\n'.encode())
        body = b''.join(parts)
        req = urllib.request.Request(self.base + '/' + method, data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        with urllib.request.urlopen(req, timeout=180) as resp:
            obj = json.loads(resp.read().decode('utf-8', errors='replace'))
        if not obj.get('ok'):
            raise RuntimeError(str(obj.get('description') or 'Telegram file upload failed'))


if QtCore is not None:
    class TelegramBridgeThread(QtCore.QThread):
        incoming = QtCore.Signal(object)
        statusChanged = QtCore.Signal(str)

        def __init__(self, root: str, token: str, allowed_user_ids: Iterable[str], parent=None):
            super().__init__(parent)
            self.root = _root_path(root)
            self.token = str(token or '').strip()
            self.allowed = {str(x).strip() for x in allowed_user_ids if str(x).strip()}
            self._stop = False
            self._outbox: 'queue.Queue[tuple]' = queue.Queue()
            self._offset = 0
            self._api: Optional[TelegramApi] = None
            self._watches: List[Dict[str, Any]] = []

        def stop(self) -> None:
            self._stop = True

        def send_text(self, chat_id: str, text: str) -> None:
            self._outbox.put(('text', str(chat_id), str(text), ''))

        def send_file(self, chat_id: str, path: str, caption: str = '') -> None:
            self._outbox.put(('file', str(chat_id), str(path), str(caption)))

        def watch_result(self, chat_id: str, model: str = '', mode: str = '', output_path: str = '', queued_at: float = 0.0) -> None:
            self._outbox.put(('watch', str(chat_id), json.dumps({
                'model': str(model or ''), 'mode': str(mode or ''),
                'output_path': str(output_path or ''), 'queued_at': float(queued_at or time.time())
            }), ''))

        def _watch_dirs(self, model: str, mode: str) -> List[Path]:
            m = str(model or '').lower()
            specific: List[Path] = []
            if m == 'zimage_gguf': specific = [self.root / 'output' / 'photo' / 'txt2img']
            elif m == 'lens': specific = [self.root / 'output' / 'lens_turbo_u4']
            elif m == 'chroma': specific = [self.root / 'output' / 'images' / 'chroma']
            elif m == 'krea2': specific = [self.root / 'output' / 'images' / 'krea2']
            elif m == 'flux_klein': specific = [self.root / 'output' / 'edits' / 'flux_klein']
            elif m == 'hidream': specific = [self.root / 'output' / 'hidream']
            elif m == 'ltx23': specific = [self.root / 'output' / 'video' / 'ltx23']
            elif m == 'ltx25': specific = [self.root / 'output' / 'video']
            elif m == 'minimax_h3': specific = [self.root / 'output' / 'video' / 'minimax_h3']
            # Always include the full output tree as a fallback. Several FrameVision
            # helpers use configurable output folders, so a hard-coded model folder
            # alone can miss a perfectly completed job.
            specific.append(self.root / 'output')
            seen = set(); out = []
            for d in specific:
                key = str(d.resolve())
                if key not in seen:
                    seen.add(key); out.append(d)
            return out

        def _snapshot_files(self, dirs: List[Path]) -> set[str]:
            found: set[str] = set()
            exts = _IMAGE_EXTS | _VIDEO_EXTS | _AUDIO_EXTS
            for d in dirs:
                if not d.exists(): continue
                try:
                    for p in d.rglob('*'):
                        if p.is_file() and p.suffix.lower() in exts:
                            found.add(str(p.resolve()))
                except Exception:
                    pass
            return found

        def _poll_result_watches(self) -> None:
            if self._api is None or not self._watches:
                return
            now = time.time(); keep = []
            for w in self._watches:
                try:
                    exact = Path(str(w.get('output_path') or '')) if w.get('output_path') else None
                    candidate = None
                    if exact is not None and exact.exists() and exact.is_file():
                        candidate = exact
                    else:
                        dirs = list(w.get('dirs') or [])
                        baseline = set(w.get('baseline') or [])
                        fresh = []
                        for d in dirs:
                            dp = Path(d)
                            if not dp.exists(): continue
                            for p in dp.rglob('*'):
                                try:
                                    if not (p.is_file() and p.suffix.lower() in (_IMAGE_EXTS | _VIDEO_EXTS | _AUDIO_EXTS)):
                                        continue
                                    rp = str(p.resolve())
                                    # Prefer genuinely new files. The timestamp fallback catches
                                    # outputs created after the watch was registered even if a
                                    # generator writes into a configurable directory.
                                    if rp not in baseline and p.stat().st_mtime >= float(w.get('queued_at') or 0) - 5:
                                        fresh.append(p)
                                except Exception:
                                    pass
                        if fresh:
                            candidate = max(fresh, key=lambda p: p.stat().st_mtime)
                        elif now - float(w.get('created_at') or now) >= 2.0:
                            # Race-safe fallback: the worker may create the result between queueing
                            # and registration of the watch, making it part of the baseline. Look
                            # for the newest media file created after the queue request anyway.
                            rescue = []
                            output_root = self.root / 'output'
                            if output_root.exists():
                                try:
                                    for p in output_root.rglob('*'):
                                        if p.is_file() and p.suffix.lower() in (_IMAGE_EXTS | _VIDEO_EXTS | _AUDIO_EXTS):
                                            if p.stat().st_mtime >= float(w.get('queued_at') or 0) - 1:
                                                rescue.append(p)
                                except Exception:
                                    pass
                            if rescue:
                                candidate = max(rescue, key=lambda p: p.stat().st_mtime)
                    if candidate is not None:
                        # Avoid uploading a file while the generator is still writing it.
                        size = candidate.stat().st_size
                        if size > 0 and size == int(w.get('last_size') or -1):
                            self._api.send_file(str(w['chat_id']), candidate, f'FrameVision result: {candidate.name}')
                            continue
                        w['last_size'] = size
                    if now - float(w.get('created_at') or now) < 6 * 3600:
                        keep.append(w)
                except Exception as exc:
                    # Keep the watch alive so a temporary Telegram/network/file-lock
                    # problem can recover on the next poll, but expose the error in
                    # FrameVision instead of failing silently.
                    try:
                        self.statusChanged.emit(f'Telegram result watcher: {exc}')
                    except Exception:
                        pass
                    w['last_size'] = -1
                    keep.append(w)
            self._watches = keep

        def _flush_outbox(self) -> None:
            if self._api is None:
                return
            for _ in range(20):
                try:
                    kind, chat_id, payload, caption = self._outbox.get_nowait()
                except queue.Empty:
                    break
                try:
                    if kind == 'file':
                        self._api.send_file(chat_id, Path(payload), caption)
                    elif kind == 'watch':
                        info = json.loads(payload or '{}')
                        dirs = self._watch_dirs(str(info.get('model') or ''), str(info.get('mode') or ''))
                        self._watches.append({
                            'chat_id': chat_id, 'model': str(info.get('model') or ''), 'mode': str(info.get('mode') or ''),
                            'output_path': str(info.get('output_path') or ''), 'queued_at': float(info.get('queued_at') or time.time()),
                            'dirs': [str(x) for x in dirs], 'baseline': list(self._snapshot_files(dirs)),
                            'created_at': time.time(), 'last_size': -1,
                        })
                    else:
                        self._api.send_message(chat_id, payload)
                except Exception as exc:
                    self.statusChanged.emit(f'Telegram send error: {exc}')

        def _allowed(self, message: Dict[str, Any]) -> bool:
            user = dict(message.get('from') or {})
            uid = str(user.get('id') or '')
            # Empty whitelist is intentionally deny-all.
            return bool(uid and uid in self.allowed)

        def _attachment_from_message(self, message: Dict[str, Any], chat_id: str) -> List[Dict[str, str]]:
            if self._api is None:
                return []
            item = None
            kind = ''
            filename = ''
            if message.get('photo'):
                photos = list(message.get('photo') or [])
                item = photos[-1] if photos else None
                kind = 'image'; filename = 'photo.jpg'
            elif message.get('video'):
                item = dict(message.get('video') or {}); kind = 'video'; filename = str(item.get('file_name') or 'video.mp4')
            elif message.get('audio'):
                item = dict(message.get('audio') or {}); kind = 'audio'; filename = str(item.get('file_name') or 'audio.mp3')
            elif message.get('voice'):
                item = dict(message.get('voice') or {}); kind = 'audio'; filename = 'voice.ogg'
            elif message.get('document'):
                item = dict(message.get('document') or {}); filename = str(item.get('file_name') or 'document.bin')
                ext = Path(filename).suffix.lower()
                kind = 'image' if ext in _IMAGE_EXTS else 'video' if ext in _VIDEO_EXTS else 'audio' if ext in _AUDIO_EXTS else 'file'
            if not item:
                return []
            fid = str(item.get('file_id') or '')
            if not fid:
                return []
            remote = self._api.get_file_path(fid)
            if not remote:
                return []
            safe = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(filename).name)[:120] or ('upload' + Path(remote).suffix)
            dest = self.root / 'temp' / 'telegram' / str(chat_id) / f'{int(time.time())}_{uuid.uuid4().hex[:6]}_{safe}'
            self._api.download_file(remote, dest)
            return [{'kind': kind, 'path': str(dest.resolve()), 'name': safe}]

        def run(self) -> None:
            if not self.token:
                self.statusChanged.emit('Telegram disabled: bot token is empty.')
                return
            if not self.allowed:
                self.statusChanged.emit('Telegram disabled: add at least one allowed Telegram user ID.')
                return
            try:
                self._api = TelegramApi(self.token)
                me = self._api.get_me()
                self.statusChanged.emit('Telegram connected as @' + str(me.get('username') or me.get('first_name') or 'bot'))
            except Exception as exc:
                self.statusChanged.emit(f'Telegram connection failed: {exc}')
                return
            while not self._stop:
                try:
                    self._flush_outbox()
                    self._poll_result_watches()
                    updates = self._api.get_updates(self._offset, 5)
                    for upd in updates:
                        self._offset = max(self._offset, int(upd.get('update_id') or 0) + 1)
                        msg = dict(upd.get('message') or {})
                        if not msg:
                            continue
                        chat_id = str((msg.get('chat') or {}).get('id') or '')
                        user_id = str((msg.get('from') or {}).get('id') or '')
                        if not self._allowed(msg):
                            # Do not disclose functionality to unapproved accounts.
                            continue
                        text = str(msg.get('text') or msg.get('caption') or '').strip()
                        attachments = self._attachment_from_message(msg, chat_id)
                        self.incoming.emit({'chat_id': chat_id, 'user_id': user_id, 'text': text, 'attachments': attachments})
                    self._flush_outbox()
                    self._poll_result_watches()
                except Exception as exc:
                    self.statusChanged.emit(f'Telegram polling error: {exc}')
                    for _ in range(10):
                        if self._stop:
                            break
                        time.sleep(0.5)
            try:
                self._flush_outbox()
            except Exception:
                pass
            self.statusChanged.emit('Telegram stopped.')
else:
    TelegramBridgeThread = None  # type: ignore
