from __future__ import annotations

import sys
from pathlib import Path


def _load_main():
    """Load timeline_editor whether launched from FrameVision root or helpers/."""
    this_file = Path(__file__).resolve()
    root = this_file.parents[1]
    helpers_dir = this_file.parent

    for path in (str(root), str(helpers_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        from helpers.timeline_editor import main
        return main
    except Exception:
        from timeline_editor import main
        return main


if __name__ == "__main__":
    raise SystemExit(_load_main()())
