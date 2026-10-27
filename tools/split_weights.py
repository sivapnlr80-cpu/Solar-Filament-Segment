"""Split checkpoints into parts below GitHub's 100 MB file limit and write weights/MANIFEST.json.

    python tools/split_weights.py maskrcnn_allcat_fold0.pth refiner.pth
"""
import hashlib
import json
import sys
from pathlib import Path

CHUNK = 50_000_000  # bytes; below GitHub's 50 MB warning threshold
WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def split(path):
    data = Path(path).read_bytes()
    parts = []
    for i in range(0, len(data), CHUNK):
        chunk = data[i:i + CHUNK]
        name = f"{Path(path).name}.part{i // CHUNK:02d}"
        (WEIGHTS / name).write_bytes(chunk)
        parts.append({"name": name, "size": len(chunk), "sha256": sha256(chunk)})
    return {"size": len(data), "sha256": sha256(data), "parts": parts}


if __name__ == "__main__":
    WEIGHTS.mkdir(exist_ok=True)
    manifest = {"chunk_size": CHUNK, "files": {Path(p).name: split(p) for p in sys.argv[1:]}}
    (WEIGHTS / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for name, entry in manifest["files"].items():
        print(name, entry["size"], "bytes ->", len(entry["parts"]), "parts, sha256", entry["sha256"])
