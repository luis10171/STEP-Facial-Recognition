"""Launch STEP without opening a camera or database at import time."""
import argparse
import sqlite3
import sys


def main():
    parser = argparse.ArgumentParser(description="STEP Medical Database and Facial Scanner")
    parser.add_argument("--demo", action="store_true", help="Use fictional records in an in-memory database")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--data-dir", help="Override the local database directory")
    parser.add_argument("--check", action="store_true", help="Check dependencies and models without opening the camera")
    args = parser.parse_args()
    try:
        if args.check:
            import importlib.metadata
            import tkinter
            from step.recognition import FaceEngine
            for name in ("PySimpleGUI", "opencv-python", "numpy"):
                print(f"{name}: {importlib.metadata.version(name)}")
            print(f"Tk: {tkinter.TkVersion}")
            FaceEngine()
            print("Recognition models verified and loaded. Camera not opened.")
        else:
            from step.app import Application
            Application(demo=args.demo, camera_index=args.camera, data_dir=args.data_dir).run()
    except (ImportError, OSError, RuntimeError, sqlite3.Error) as exc:
        print(f"STEP could not start: {exc}\nSee README.md for setup and troubleshooting.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
