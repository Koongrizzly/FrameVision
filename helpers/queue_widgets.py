
from __future__ import annotations
import json, os, statistics, time
from pathlib import Path
from datetime import datetime
from PySide6.QtCore import Qt, QSize, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QProgressBar, QToolButton

DT_FMT = "%Y-%m-%d %H:%M:%S"

def _parse_dt(s: str):
    if not s: return None
    try:
        return datetime.strptime(s, DT_FMT)
    except Exception:
        return None

def _fmt_dur(secs: float) -> str:
    try:
        secs = int(max(0, secs)); m,s = divmod(secs,60); h,m = divmod(m,60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
    except Exception: return "0:00"

class JobRowWidget(QWidget):
    """Row with filename, progress, and explicit Added / ETA / Ended labels. Theme colors respected."""
    def __init__(self, job_path: str, status: str):
        super().__init__()
        self.job_path = Path(job_path)
        self.status = status  # pending/running/done/failed
        try:
            self.data = json.loads(self.job_path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}
        self._build()

    def _build(self):
        d = self.data or {}
        # Title: filename + type
        base = os.path.basename(d.get("input","")) or "Unknown"
        jtype = d.get("type","?")
        title = f"{base}   ·   {(d.get('title') or (d.get('args',{}) or {}).get('label', jtype)).capitalize()}"
        self.title = QLabel(title); self.title.setObjectName("jobRowTitle"); self.title.setWordWrap(False); self.title.setToolTip(title)

        # meta lines
        self.meta = QLabel(self._build_meta()); self.meta.setObjectName("jobRowMeta"); self.meta.setWordWrap(False); self.meta.setToolTip(self.meta.text())

        # Progress bar
        self.bar = QProgressBar(); self.bar.setMinimumHeight(6); self.bar.setTextVisible(False)
        if self.status == "running": self.bar.setRange(0,0)
        elif self.status == "done": self.bar.setRange(0,100); self.bar.setValue(100)
        elif self.status == "failed": self.bar.setRange(0,100); self.bar.setValue(100); self.bar.setStyleSheet("QProgressBar::chunk { background-color:#ff4d4f; }")
        else: self.bar.setRange(0,100); self.bar.setValue(0)

        # Open button for finished/failed
        self.btn_open = QToolButton(); self.btn_open.setText("Open"); self.btn_open.setToolTip("Open output folder")
        self.btn_open.clicked.connect(self._open_folder)
        self.btn_open.setVisible(self.status in ("done","failed"))

        top = QHBoxLayout(); top.setContentsMargins(0,0,0,0); top.setSpacing(6); top.addWidget(self.title, 1); top.addWidget(self.btn_open, 0)

        lay = QVBoxLayout(self); lay.setContentsMargins(8,6,8,6); lay.setSpacing(4)
        lay.addLayout(top); lay.addWidget(self.meta); lay.addWidget(self.bar)

        if self.status == "running":
            self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(1000)

    # --- Helpers
    def _avg_recent(self)->float|None:
        # Median of up to 20 completed jobs of same type
        try:
            done = self.job_path.parent.parent / "done"
            t = (self.data or {}).get("type","")
            label = (self.data or {}).get("title") or ((self.data or {}).get("args",{}) or {}).get("label","")
            durations=[]
            for p in sorted(done.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    if d.get("type")==t and ((d.get("title") or (d.get("args",{}) or {}).get("label",""))==label or not label):
                        dur = d.get("duration_sec", None)
                        if isinstance(dur,(int,float)) and dur>0:
                            durations.append(dur); continue
                        # fallback: file mtime - created_at
                        ca = _parse_dt(d.get("created_at",""))
                        if ca:
                            dur2 = max(0, int(p.stat().st_mtime - ca.timestamp()))
                            durations.append(dur2)
                except Exception: pass
            return statistics.median(durations) if durations else None
        except Exception:
            return None

    def _open_folder(self):
        d=self.data or {}; args=d.get("args",{})
        path = args.get("outfile","") or d.get("out_dir","") or ""
        if path:
            p = Path(path).expanduser(); folder = p if p.is_dir() else p.parent
            if folder.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _build_meta(self)->str:
        d = self.data or {}
        created = _parse_dt(d.get("created_at",""))
        started = _parse_dt(d.get("started_at",""))
        finished = _parse_dt(d.get("finished_at",""))
        # Fallbacks if worker didn't stamp times
        if not finished and self.status in ("done","failed"):
            try:
                finished = datetime.fromtimestamp(self.job_path.stat().st_mtime)
            except Exception:
                pass
        if not started and self.status in ("running","done","failed"):
            started = created or started

        # Compute elapsed/eta/duration
        now = datetime.now()
        elapsed = int((now - (started or created or now)).total_seconds()) if self.status=="running" else None
        avg = self._avg_recent() if self.status=="running" else None
        eta = max(0, int((avg - elapsed))) if (avg and elapsed is not None) else None
        took = None
        if finished and (started or created):
            took = int((finished - (started or created)).total_seconds())

        added_s   = created.strftime(DT_FMT) if created else "—"
        eta_s     = _fmt_dur(eta) if eta is not None else "—"
        ended_s   = finished.strftime(DT_FMT) if finished else "—"
        extra = ""
        if self.status == "running":
            extra = f"  ·  elapsed { _fmt_dur(elapsed or 0) }"
        if self.status in ("done","failed") and took is not None:
            extra = f"  ·  took { _fmt_dur(took) }"
        err = str((self.data or {}).get("error","")).strip()
        if self.status == "failed" and err:
            extra = f"{extra}  ·  error: {err}"
        return f"Added: {added_s}    |    Time left: {eta_s}    |    Ended: {ended_s}{extra}"

    def _tick(self):
        self.meta.setText(self._build_meta())

    def sizeHint(self)->QSize:
        return QSize(220, 44)
