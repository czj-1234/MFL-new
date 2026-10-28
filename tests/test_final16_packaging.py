"""Offline smoke test: compact export keeps analysis data and no model tensors."""
from __future__ import annotations
import csv
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Final16PackagingTests(unittest.TestCase):
    def test_server_a_compressed_export_excludes_large_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "scripts/acm_revision"
            folder.mkdir(parents=True)
            script = folder / "package_final16_analysis.sh"
            shutil.copy2(ROOT / "scripts/acm_revision/package_final16_analysis.sh", script)
            run = (root / "results/acm_revision/final16_20261022/uncentered/"
                   "hateful_memes__clip_dual__target__modality_exclusive__c0.7"
                   "__n12__p1.0__fedavg__seed42__def-layerwise_mixed_filter")
            run.mkdir(parents=True)
            (run / "summary.json").write_text(json.dumps({"rounds": 150}))
            (run / "capture_manifest.json").write_text(json.dumps({"status": "PASS"}))
            (run / "round_metrics.csv").write_text("round,val_auroc\n1,0.5\n150,0.66\n")
            (run / "update_metadata.csv").write_text(
                "round,defense_l2_full_post\n150,0.04\n")
            (run / "resolved_config.json").write_text("{}")
            (run / "best_model.pt").write_bytes(b"n" * 120_000)
            (run / "updates").mkdir()
            (run / "updates/delta.npz").write_bytes(b"n" * 120_000)
            (run / "client_partition_manifest.json").write_text(
                '{"sensitive_sample_ids": [1234]}')
            audit = root / "results/acm_revision/final16_20261022/audits"
            audit.mkdir(parents=True)
            (audit / "uncentered_statistics.json").write_text('{"mean": 0.5}')
            p = subprocess.run(["bash", str(script), "A"], cwd=root,
                               text=True, capture_output=True, check=True)
            self.assertIn("[STATUS] 1/8", p.stdout)
            archives = list((root / "exports").glob("*.tar.gz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0]) as tar:
                names = tar.getnames()
                self.assertTrue(any(x.endswith("update_metadata.csv") for x in names))
                self.assertTrue(any(x.endswith("uncentered_statistics.json") for x in names))
                self.assertFalse(any(x.endswith(".pt") or x.endswith(".npz")
                                     for x in names))
                self.assertFalse(any("client_partition_manifest.json" in x
                                     for x in names))
                rows = list(csv.DictReader(io.StringIO(
                    tar.extractfile("META/RUN_STATUS.csv").read().decode())))
                self.assertEqual(len(rows), 8)
                self.assertEqual(rows[3]["seed"], "42")
                self.assertEqual(rows[3]["phase"], "uncentered")
                self.assertEqual(rows[3]["status"], "PASS")
                self.assertEqual(rows[4]["status"], "MISSING")

    def test_shared_storage_combined_archive_contains_both_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "scripts/acm_revision"
            folder.mkdir(parents=True)
            script = folder / "package_final16_analysis.sh"
            shutil.copy2(ROOT / "scripts/acm_revision/package_final16_analysis.sh", script)
            base = root / "results/acm_revision/final16_20261022"
            for phase, concentration in (("uncentered", "0.7"), ("c09", "0.9")):
                run = (base / phase /
                       f"hateful_memes__clip_dual__target__modality_exclusive__c{concentration}"
                       "__n12__p1.0__fedavg__seed42__def-layerwise_mixed_filter")
                run.mkdir(parents=True)
                (run / "summary.json").write_text('{"rounds": 150}')
                (run / "capture_manifest.json").write_text('{"status": "PASS"}')
                (run / "round_metrics.csv").write_text("round,val_auroc\n150,0.65\n")
                (run / "update_metadata.csv").write_text("round,defense_l2_full_post\n150,0.04\n")
                (run / "best_model.pt").write_bytes(b"secret" * 1000)
            audit = base / "audits"
            audit.mkdir()
            (audit / "uncentered_statistics.json").write_text('{"value": 1}')
            (audit / "c09_statistics.json").write_text('{"value": 2}')
            result = subprocess.run(["bash", str(script), "ALL"], cwd=root,
                                    text=True, capture_output=True, check=True)
            self.assertIn("[STATUS] 2/16", result.stdout)
            self.assertIn("[PHASE uncentered] 1/8", result.stdout)
            self.assertIn("[PHASE c09] 1/8", result.stdout)
            archives = list((root / "exports").glob("aslwf_final16_combined_*_analysis.tar.gz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0]) as tar:
                names = tar.getnames()
                self.assertTrue(any("/uncentered/" in name and name.endswith("summary.json")
                                    for name in names))
                self.assertTrue(any("/c09/" in name and name.endswith("summary.json")
                                    for name in names))
                self.assertTrue(any(name.endswith("/uncentered_statistics.json") for name in names))
                self.assertTrue(any(name.endswith("/c09_statistics.json") for name in names))
                self.assertFalse(any(x.endswith(".pt") for x in names))
                rows = list(csv.DictReader(io.StringIO(
                    tar.extractfile("META/RUN_STATUS.csv").read().decode())))
                self.assertEqual(len(rows), 16)
                self.assertEqual({r["phase"] for r in rows}, {"uncentered", "c09"})
                self.assertEqual(sum(r["status"] == "PASS" for r in rows), 2)

    def test_missing_phase_refuses_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "scripts/acm_revision"
            folder.mkdir(parents=True)
            script = folder / "package_final16_analysis.sh"
            shutil.copy2(ROOT / "scripts/acm_revision/package_final16_analysis.sh", script)
            p = subprocess.run(["bash", str(script), "B"], cwd=root,
                               text=True, capture_output=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("Experiment results missing", p.stderr + p.stdout)
            self.assertFalse((root / "exports").exists())


if __name__ == "__main__":
    unittest.main()
