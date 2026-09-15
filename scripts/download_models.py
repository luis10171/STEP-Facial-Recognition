"""Download pinned upstream models; verify before replacing local files."""

import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["models"]:
        target = ROOT / "models" / item["name"]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]:
            print(f"Verified {target.name}")
            continue
        print(f"Downloading {target.name} ({item['size'] / 1_000_000:.1f} MB)...", flush=True)
        request = urllib.request.Request(item["url"], headers={"User-Agent": "STEP-portfolio-demo"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(item["size"] + 1)
        if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Integrity check failed for {target.name}; existing file was kept.")
        temporary = target.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(target)
        print(f"Verified {target.name}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
