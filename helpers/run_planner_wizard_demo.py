# helpers/run_planner_wizard_demo.py
from __future__ import annotations

import os
import sys

# Allow running both as:
#   python -m helpers.run_planner_wizard_demo
# and:
#   python helpers/run_planner_wizard_demo.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PySide6 import QtWidgets  # type: ignore

try:
    from helpers.planner_wizard_placeholder import PlannerWizard
except Exception:
    # Fallback when running from within a package context
    from .planner_wizard_placeholder import PlannerWizard  # type: ignore


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = PlannerWizard()
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
