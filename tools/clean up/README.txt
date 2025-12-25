Repo Sanity Kit (fixed)
-----------------------
If you have a venv at .\.venv it will use it. Otherwise it will try your system Python.
Run:
  tools\diagnostics\run_repo_sanity.bat
Outputs are written to your project root:
  - repo_sanity_report.md
  - repo_sanity_unused.csv
  - repo_sanity_duplicates.csv
  - repo_sanity_quarantine.bat
