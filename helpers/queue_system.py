
from __future__ import annotations
import os, sys, subprocess, time
from pathlib import Path
from typing import Tuple

class QueueSystem:
    def __init__(self, base: Path | str):
        self.base = Path(base)
        self.jobs = {
            "pending": self.base / "jobs" / "pending",
            "running": self.base / "jobs" / "running",
            "done":    self.base / "jobs" / "done",
            "failed":  self.base / "jobs" / "failed",
        }
        self.logs = self.base / "logs"
        for p in list(self.jobs.values()) + [self.logs]:
            p.mkdir(parents=True, exist_ok=True)

    @property
    def heartbeat_path(self)->Path:
        return self.logs / "worker_heartbeat.txt"

    def heartbeat_recent(self, max_age: float = 5.0) -> bool:
        try:
            if self.heartbeat_path.exists():
                return (time.time() - self.heartbeat_path.stat().st_mtime) < max_age
        except Exception:
            pass
        return False

    # --- Worker ---
    def start_worker(self):
        py = sys.executable or "python"
        # Prefer pythonw.exe to avoid a console window on Windows
        if os.name == "nt" and sys.executable:
            try:
                pyw = Path(sys.executable).with_name("pythonw.exe")
                if pyw.exists():
                    py = str(pyw)
            except Exception:
                pass
        worker = self.base / "helpers" / "worker.py"
        if os.name == "nt":
            CREATE_NO_WINDOW = 0x08000000
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            subprocess.Popen([py, str(worker)], creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, close_fds=True)
        else:
            subprocess.Popen([py, str(worker)], start_new_session=True, close_fds=True)

    # --- Ops ---
    def remove_failed(self) -> int:
        count = 0
        for p in list(self.jobs["failed"].glob("*.json")):
            try:
                p.unlink(); count += 1
            except Exception:
                pass
        return count

    def clear_finished_failed(self) -> Tuple[int,int]:
        c1=c2=0
        for p in list(self.jobs["done"].glob("*.json")):
            try: p.unlink(); c1 += 1
            except Exception: pass
        for p in list(self.jobs["failed"].glob("*.json")):
            try: p.unlink(); c2 += 1
            except Exception: pass
        return c1, c2

    def recover_running_to_pending(self) -> int:
        moved = 0
        for p in sorted(self.jobs["running"].glob("*.json")):
            target = self.jobs["pending"] / p.name
            try:
                p.replace(target); moved += 1
            except Exception:
                pass
        return moved

    def mark_running_failed(self) -> int:
        moved = 0
        for p in sorted(self.jobs["running"].glob("*.json")):
            target = self.jobs["failed"] / p.name
            try:
                p.replace(target); moved += 1
            except Exception:
                pass
        return moved

    def cancel_pending(self, name: str):
        p = self.jobs["pending"] / name
        if p.exists():
            p.unlink()

    def cancel_running_by_jobid(self, jid: str):
        cancel = self.jobs["running"] / f"{jid}.cancel"
        cancel.write_text("cancel", encoding="utf-8")

    def nudge_pending(self, name: str, delta: int):
        p = self.jobs["pending"] / name
        if not p.exists():
            return
        try:
            st = p.stat().st_mtime
            os.utime(p, (st+delta, st+delta))
        except Exception:
            pass

def _autostart_if_enabled():
    try:
        # Skip if we're in the worker process
        if os.environ.get('FRAMEVISION_WORKER') == '1':
            return
        s = QSettings('FrameVision','FrameVision')
        enabled = s.value('worker/auto_start', None)
        if enabled is None:
            # Backward-compat: respect old rife toggle, default ON
            enabled = s.value('rife/auto_start_worker', 1)
        enabled = int(enabled) if enabled is not None else 1
        if enabled:
            qs = QueueSystem(Path('.'))
            try:
                alive = qs.heartbeat_recent(5.0)
            except Exception:
                alive = False
            if not alive:
                # Start quietly
                qs.start_worker()
    except Exception:
        pass

# Trigger autostart as soon as this module is imported (GUI loads this early)
try:
    _autostart_if_enabled()
except Exception:
    pass
