"""The model checkpoints are stored in the repository as parts of at most 50 MB (GitHub rejects files over 100 MB).

``assemble`` concatenates the parts listed in ``weights/MANIFEST.json`` into the original ``.pth`` file and verifies every
part and the final file by SHA-256. The assembled files go to ``weights/assembled/`` (git-ignored).
"""
import hashlib
import json
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def assemble(name, weights_dir=WEIGHTS_DIR, out_dir=None):
    weights_dir = Path(weights_dir)
    entry = json.loads((weights_dir / "MANIFEST.json").read_text(encoding="utf-8"))["files"][name]
    target = Path(out_dir or weights_dir / "assembled") / name
    if target.exists() and _sha256(target.read_bytes()) == entry["sha256"]:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    for part in entry["parts"]:
        chunk = (weights_dir / part["name"]).read_bytes()
        if _sha256(chunk) != part["sha256"]:
            raise ValueError(f"{part['name']}: checksum mismatch")
        blob += chunk
    if _sha256(blob) != entry["sha256"]:
        raise ValueError(f"{name}: checksum mismatch after assembly")
    target.write_bytes(blob)
    return target
