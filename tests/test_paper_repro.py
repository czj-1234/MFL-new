"""CPU-only reproducibility guard tests. No GPU or dataset access required."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "acm_revision" / "paper_repro_preflight.py"
spec = importlib.util.spec_from_file_location("paper_repro_preflight", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class PaperReproGuards(unittest.TestCase):
    def test_canonical_rounds_and_attack_count(self):
        self.assertEqual(module.PAPER_ROUNDS, (1, 10, 30, 50, 75, 100, 125, 150))
        self.assertEqual(len(module.EXPECTED_GROUPS) * 3 * 2, 54)
        self.assertEqual(module.EXPECTED_TARGET_SEEDS, (42, 43, 44, 45, 46))

    def test_code_lineage_and_missing_artifacts_are_not_a_pass(self):
        report = module.run(REPO_ROOT)
        self.assertNotEqual(report["status"], "FAIL", json.dumps(report, indent=2))
        self.assertIn(report["status"], ("NEEDS_ARTIFACTS", "REVIEW_REQUIRED", "PASS"))
        self.assertTrue(any(x["check"] == "eight-milestone-paper-audit-available"
                            and x["status"] == "PASS" for x in report["findings"]))
        # A code-only checkout is NOT proof of numerical reproducibility.
        if not (REPO_ROOT / module.HEAD_MANIFEST).exists():
            self.assertNotEqual(report["status"], "PASS")

    def test_strict_mode_rejects_missing_results(self):
        if not (REPO_ROOT / module.HEAD_MANIFEST).exists():
            report = module.run(REPO_ROOT, strict=True)
            self.assertEqual(report["status"], "FAIL")
            self.assertGreater(report["counts"]["missing"], 0)

    def test_paper_milestone_filter_and_distinct_target_metrics_exist(self):
        content = (REPO_ROOT / "scripts" / "acm_revision" / "audit_headfix.py").read_text()
        for required in (
            "--paper-milestones", "PAPER_MILESTONE_ROUNDS",
            "paper_milestones", "target_pooled_auc_diagnostic",
            "target_mean_per_independent_run_auroc",
            'metadata["round"].astype(int).isin(PAPER_MILESTONE_ROUNDS)',
        ):
            self.assertIn(required, content)

    def test_canonical_job_generator_is_non_destructive_and_milestone_locked(self):
        plan = (REPO_ROOT / "src" / "acm_revision" / "paper_protocol_plan.py").read_text()
        for marker in (
            "PAPER_MILESTONES = (1, 10, 30, 50, 75, 100, 125, 150)",
            "PAPER_FULL_UPDATE_DIM = 2048",
            '"historical_outputs_untouched": True',
            '"PLANNED_NOT_RUN"',
            'if list(directory.glob("*.yaml")):',
            '"core", "baselines", "complete"',
        ):
            self.assertIn(marker, plan)

    def test_table7_comparator_checks_distinct_metrics_and_five_runs(self):
        path = REPO_ROOT / "scripts" / "acm_revision" / "verify_paper_table7.py"
        source = path.read_text()
        for marker in ("EXPECTED_TABLE7", "TARGETS = (42, 43, 44, 45, 46)",
                       "all_shared_cpt_diag_max", "head_cpt_rf",
                       "shadow_selected", "strict-protocol",
                       'if candidate_count != 54:'):
            self.assertIn(marker, source)

    def test_appendix_traceability_covers_main_results(self):
        content = (REPO_ROOT / "docs" / "paper_repro" / "TRACEABILITY.csv").read_text()
        for table in ("Table 1", "Table 2", "Table 3", "Table 4",
                      "Table 5", "Table 6", "Table 7", "Table 8",
                      "Table D1", "Table E3", "Table F1"):
            self.assertIn(table, content)


if __name__ == "__main__":
    unittest.main()
