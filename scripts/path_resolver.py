import os, json, configparser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _norm(p):
    if not p: return None
    p = os.path.expandvars(os.path.expanduser(p))
    if not os.path.isabs(p):
        p = os.path.join(ROOT, p)
    return os.path.normpath(p)

def _try_json(paths):
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            obj = data.get("paths", data)
            return _norm(obj.get("models_dir")), _norm(obj.get("bin_dir"))
        except Exception:
            pass
    return None, None

def _try_ini(paths):
    cp = configparser.ConfigParser()
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                cp.read_file(f)
            if cp.has_section("paths"):
                md = _norm(cp.get("paths","models_dir", fallback=None))
                bd = _norm(cp.get("paths","bin_dir", fallback=None))
                return md, bd
        except Exception:
            pass
    return None, None

def resolve():
    md = _norm(os.environ.get("FRAMEVISION_MODELS_DIR"))
    bd = _norm(os.environ.get("FRAMEVISION_BIN_DIR"))

    json_md, json_bd = _try_json([
        os.path.join(ROOT, "config", "framevision_paths.json"),
        os.path.join(ROOT, "framevision_paths.json"),
        os.path.join(ROOT, "config", "settings.json"),
        os.path.join(ROOT, "settings.json"),
        os.path.join(ROOT, "framevision_settings.json"),
    ])
    md = md or json_md
    bd = bd or json_bd

    ini_md, ini_bd = _try_ini([
        os.path.join(ROOT, "config", "framevision.ini"),
        os.path.join(ROOT, "framevision.ini"),
        os.path.join(ROOT, "config.ini"),
    ])
    md = md or ini_md
    bd = bd or ini_bd

    if not md: md = os.path.join(ROOT, "models")
    if not bd: bd = os.path.join(ROOT, "bin")

    os.makedirs(md, exist_ok=True)
    os.makedirs(bd, exist_ok=True)
    return md, bd

MODELS_DIR, BIN_DIR = resolve()
