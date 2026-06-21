import os, sys, ast, csv, time
from pathlib import Path
from collections import defaultdict, deque

APP_ROOT = Path(__file__).resolve().parents[2]
HELPERS = APP_ROOT / "helpers"
ENTRYPOINTS = [
    APP_ROOT / "framevision_run.py",
    HELPERS / "framevision_app.py",
    HELPERS / "worker.py",
]
IGNORE_DIRS = {".venv", "__pycache__", ".git", "models", "logs", "tools_quarantine", "helpers_quarantine"}

def gather_py_files(base: Path):
    files = []
    for p in base.rglob("*.py"):
        rel = p.relative_to(APP_ROOT)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        files.append(p)
    return files

def module_name_from_path(p: Path) -> str:
    rel = p.relative_to(APP_ROOT).with_suffix("")
    return ".".join(rel.parts)

def parse_imports(p: Path):
    mods = set()
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src, filename=str(p))
    except Exception:
        return mods
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                mods.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods

def has_heavy_top_level_code(p: Path) -> bool:
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src, filename=str(p))
    except Exception:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.If)):
            if isinstance(node, ast.If):
                try:
                    if isinstance(node.test, ast.Compare) and getattr(node.test.left, "id", "") == "__name__":
                        continue
                except Exception:
                    pass
            continue
        return True
    return False

def build_graph(py_files):
    name_to_path = {}
    for p in py_files:
        name_to_path[module_name_from_path(p)] = p

    graph = defaultdict(set)
    for mod, path in name_to_path.items():
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src, filename=str(path))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    base = n.name.split(".")[0]
                    if base == "helpers":
                        if n.name in name_to_path:
                            graph[mod].add(n.name)
                    elif n.name in name_to_path:
                        graph[mod].add(n.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = node.module.split(".")[0]
                if base == "helpers":
                    if node.module in name_to_path:
                        graph[mod].add(node.module)
                elif node.module in name_to_path:
                    graph[mod].add(node.module)
    return graph, name_to_path

def reachable_from(entry_mods, graph):
    seen = set()
    q = deque(entry_mods)
    while q:
        m = q.popleft()
        if m in seen:
            continue
        seen.add(m)
        for nb in graph.get(m, ()):
            if nb not in seen:
                q.append(nb)
    return seen

def find_duplicates(py_files):
    by_stem = defaultdict(list)
    for p in py_files:
        by_stem[p.stem.lower()].append(p)
    dups = []
    for stem, files in by_stem.items():
        parents = {str(f.parent) for f in files}
        if len(files) > 1 and len(parents) > 1:
            dups.append((stem, files))
    return dups

def main():
    out_md = APP_ROOT / "repo_sanity_report.md"
    out_unused = APP_ROOT / "repo_sanity_unused.csv"
    out_dups = APP_ROOT / "repo_sanity_duplicates.csv"
    out_quarantine = APP_ROOT / "repo_sanity_quarantine.bat"

    py_files = gather_py_files(APP_ROOT)
    graph, name_to_path = build_graph(py_files)

    entry_mods = []
    for ep in ENTRYPOINTS:
        if ep.exists():
            entry_mods.append(module_name_from_path(ep))
    used = reachable_from(entry_mods, graph)

    side_effects = [p for p in py_files if has_heavy_top_level_code(p)]

    unused = []
    for mod, path in name_to_path.items():
        if not str(path).startswith(str(HELPERS)):
            continue
        if mod in used:
            continue
        unused.append(path)

    dups = find_duplicates(py_files)

    with out_unused.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["unused_helper_path"])
        for p in sorted(unused):
            w.writerow([str(p.relative_to(APP_ROOT))])

    with out_dups.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["module_stem", "path"])
        for stem, files in dups:
            for p in files:
                w.writerow([stem, str(p.relative_to(APP_ROOT))])

    with out_quarantine.open("w", encoding="utf-8") as f:
        f.write("@echo off\r\n")
        f.write("setlocal\r\n")
        f.write("cd /d %~dp0\r\n")
        f.write("if not exist helpers_quarantine mkdir helpers_quarantine\r\n")
        for p in sorted(unused):
            rel = str(p.relative_to(APP_ROOT)).replace("/", "\\")
            dst = rel.replace("helpers\\", "helpers_quarantine\\")
            f.write(f"echo Moving {rel} -> {dst}\r\n")
            f.write(f"if exist \"{rel}\" move /Y \"{rel}\" \"{dst}\"\r\n")

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Repo Sanity Report\n\n")
        f.write(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- App root: {APP_ROOT}\n")
        f.write(f"- Entry modules: {', '.join(entry_mods) if entry_mods else '(none found)'}\n")
        f.write(f"- Total .py files scanned: {len(py_files)}\n")
        f.write(f"- Unused helpers (reachable from entrypoints): {len(unused)}\n")
        f.write(f"- Files with top-level side-effects: {len(side_effects)}\n")
        f.write(f"- Duplicate/shadow-prone module stems: {len(dups)}\n\n")
        if side_effects:
            f.write("## Files with top-level side effects (import can run code!)\n")
            for p in side_effects:
                f.write(f"- {p.relative_to(APP_ROOT)}\n")
            f.write("\n")
        if dups:
            f.write("## Duplicate/shadow-prone modules\n")
            seen = set()
            for stem, files in dups:
                if stem in seen: 
                    continue
                seen.add(stem)
                f.write(f"- **{stem}**:\n")
                for p in files:
                    f.write(f"  - {p.relative_to(APP_ROOT)}\n")
            f.write("\n")
        f.write("## Next steps\n")
        f.write("- Review `repo_sanity_unused.csv` and `repo_sanity_duplicates.csv`.\n")
        f.write("- If the unused list looks right, run `repo_sanity_quarantine.bat` to move them to `helpers_quarantine/` (reversible).\n")
        f.write("- Re-run your app. If anything breaks, move the specific file back.\n")

    print("[sanity] Wrote:", out_md)
    print("[sanity] Wrote:", out_unused)
    print("[sanity] Wrote:", out_dups)
    print("[sanity] Wrote:", out_quarantine)
    print("[sanity] Done.")

if __name__ == "__main__":
    main()
