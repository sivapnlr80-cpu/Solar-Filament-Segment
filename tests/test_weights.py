"""Checkpoint split / reassembly: python -m unittest discover -s tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from filament import weights  # noqa: E402

REPO_WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


class AssembleTests(unittest.TestCase):
    def make(self, tmp, payload, corrupt=False):
        parts = []
        for i in range(0, len(payload), 10):
            chunk = payload[i:i + 10]
            name = f"x.pth.part{i // 10:02d}"
            (tmp / name).write_bytes(chunk[::-1] if corrupt and i == 10 else chunk)
            parts.append({"name": name, "size": len(chunk), "sha256": weights._sha256(chunk)})
        manifest = {"files": {"x.pth": {"size": len(payload), "sha256": weights._sha256(payload), "parts": parts}}}
        (tmp / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_roundtrip(self):
        payload = bytes(range(256)) * 3
        with tempfile.TemporaryDirectory() as d:
            self.make(Path(d), payload)
            out = weights.assemble("x.pth", weights_dir=d)
            self.assertEqual(out.read_bytes(), payload)
            self.assertEqual(weights.assemble("x.pth", weights_dir=d), out)  # second call reuses the file

    def test_corrupt_part_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(Path(d), bytes(range(100)), corrupt=True)
            with self.assertRaises(ValueError):
                weights.assemble("x.pth", weights_dir=d)


@unittest.skipUnless((REPO_WEIGHTS / "MANIFEST.json").exists(), "weights/ not present")
class CommittedWeightsTests(unittest.TestCase):
    def test_committed_parts_reassemble_to_manifest_hashes(self):
        manifest = json.loads((REPO_WEIGHTS / "MANIFEST.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            for name, entry in manifest["files"].items():
                out = weights.assemble(name, out_dir=d)
                self.assertEqual(out.stat().st_size, entry["size"])
                self.assertTrue(all(p["size"] <= 50_000_000 for p in entry["parts"]))


if __name__ == "__main__":
    unittest.main()
