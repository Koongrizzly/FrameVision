
import json, uuid
from pathlib import Path
from datetime import datetime

def make_job_json(job_type: str, input_path: str, out_dir: str, args: dict, pending_dir: str, priority: int=500):
    job = {
        "id": str(uuid.uuid4())[:8],
        "type": job_type,
        "input": input_path,
        "out_dir": out_dir,
        "args": args or {},
        "priority": int(priority),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    Path(pending_dir).mkdir(parents=True, exist_ok=True)
    fn = Path(pending_dir) / f"{job['id']}_{job_type}.json"
    fn.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return str(fn)
