import os
import sys
import subprocess


def get_root_dir() -> str:
    """
    Return the directory where this launcher lives.
    Works both when frozen by PyInstaller and when run as a plain script.
    """
    if getattr(sys, "frozen", False):
        # When frozen, sys.executable is the path to the .exe
        return os.path.dirname(sys.executable)
    # When running as a script
    return os.path.dirname(os.path.abspath(__file__))


def show_error(msg: str, title: str = "FrameVision Launcher") -> None:
    """
    Show an error message. If there's no console (PyInstaller --noconsole),
    try a MessageBox. Otherwise, print to stdout as a fallback.
    """
    try:
        import ctypes  # type: ignore[attr-defined]

        ctypes.windll.user32.MessageBoxW(0, msg, title, 0)
    except Exception:
        print(f"{title}: {msg}")


def main() -> None:
    root = get_root_dir()
    os.chdir(root)

    start_bat = os.path.join(root, "start.bat")
    if not os.path.isfile(start_bat):
        show_error(
            "Could not find start.bat next to FrameVision.exe.\n\n"
            "Make sure FrameVision.exe and start.bat are both placed in the main "
            "FrameVision folder."
        )
        return

    # Delegate all logic (install checks, worker, fixes, etc.) to start.bat.
    # This mirrors the old behavior 1:1, just with an .exe entry point.
    try:
        subprocess.Popen(
            ["cmd", "/c", "call", "start.bat"],
            cwd=root,
        )
    except Exception as exc:
        show_error(f"Failed to run start.bat:\n{exc}")


if __name__ == "__main__":
    main()
