"""Submission-branch scientific protocol guards; never substitutes for numerical reruns."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def from_path(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    obj = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(obj)
    return obj


preflight = from_path("paper_repro_preflight", "scripts/acm_revision/paper_repro_preflight.py")
budget = from_path("paper_l2_budget", "src/acm_revision/l2_budget.py")


class PaperReproGuards(unittest.TestCase):
    def test_single_paper_calendar_and_full_update_dim(self):
        self.assertEqual(preflight.PAPER_ROUNDS, (1, 10, 30, 50, 75, 100, 125, 150))
        self.assertEqual(len(preflight.EXPECTED_GROUPS) * 3 * 2, 54)
        self.assertEqual(preflight.EXPECTED_TARGET_SEEDS, (42, 43, 44, 45, 46))
        plan = (ROOT / "src/acm_revision/paper_protocol_plan.py").read_text()
        self.assertIn("PAPER_FULL_UPDATE_DIM = 2048", plan)
        capture = (ROOT / "src/acm_revision/privacy_capture_runner_v2.py").read_text()
        self.assertIn("if projection_dim != 2048:", capture)
        self.assertIn("Paper protocol requires exactly eight fixed capture rounds", capture)

    def test_code_only_is_never_result_reproduction(self):
        report = preflight.run(ROOT)
        self.assertIn(report["status"], ("NEEDS_ARTIFACTS", "PASS"), json.dumps(report, indent=2))
        if not (ROOT / preflight.HEAD_MANIFEST).exists():
            self.assertEqual(report["status"], "NEEDS_ARTIFACTS")
            strict = preflight.run(ROOT, strict=True)
            self.assertEqual(strict["status"], "FAIL")
            self.assertGreater(strict["counts"]["missing"], 0)

    def test_exactly_one_paper_attack_mode(self):
        src = (ROOT / "scripts/acm_revision/audit_headfix.py").read_text()
        self.assertNotIn('parser.add_argument("--paper-milestones"', src)
        self.assertIn('metadata["round"].astype(int).isin(PAPER_MILESTONE_ROUNDS)', src)
        self.assertIn('milestone_rounds_present', src)
        self.assertIn("target_mean_per_independent_run_auroc", src)
        self.assertIn("target_pooled_auc_diagnostic", src)

    def test_shared_l2_budget_must_be_enforced_on_all_subspace_methods(self):
        dispatcher = (ROOT / "src/acm_revision/defense_dispatch.py").read_text()
        generator = (ROOT / "src/acm_revision/defense70_plan.py").read_text()
        complete = (ROOT / "src/acm_revision/headfix_plan.py").read_text()
        self.assertIn("apply_common_l2_cap_inplace(", dispatcher)
        self.assertIn("def _enforce_shared_l2(", dispatcher)
        self.assertIn("max_relative_update_distortion_l2", generator)
        self.assertIn("max_relative_update_distortion_l2", complete)
        self.assertIn("Missing measured shadow-only L2 budget", generator)
        self.assertIn("Complete ASLWF requires the same locked strong L2 budget", complete)

    def test_l2_budget_scales_real_change(self):
        self.assertAlmostEqual(budget.scaling_factor(10, 5, .2), .4)
        self.assertAlmostEqual(budget.scaling_factor(10, 1, .2), 1.0)
        self.assertAlmostEqual(budget.scaling_factor(0, 3, .2), 0.0)
        for invalid in (-1, 0, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                budget.scaling_factor(10, 3, invalid)

    def test_config_generator_is_one_prospective_pipeline(self):
        src = (ROOT / "src/acm_revision/paper_protocol_plan.py").read_text()
        for marker in (
            "PAPER_MILESTONES = (1, 10, 30, 50, 75, 100, 125, 150)",
            "PAPER_FULL_UPDATE_DIM = 2048",
            '"PLANNED_NOT_RUN"',
            'if list(directory.glob("*.yaml")):',
            '"core", "baselines", "complete"',
            "def _core_templates(base_config: Path)",
        ):
            self.assertIn(marker, src)
        self.assertNotIn("core72_modality_first_v2_r150", src)

    def test_70_and_8_generated_formal_configs_share_locked_real_l2_caps(self):
        import tempfile
        from src.acm_revision.paper_protocol_plan import _defense_configs, _complete_configs
        base_config = ROOT / "configs/acm_revision/hateful_memes.yaml"
        basis = "frozen_sha256_named_basis.npz"  # synthetic config-only path, not numeric evidence
        matched = {
            "status": "LOCKED",
            "provenance": {"target_data_used_for_basis_or_tuning": False},
            "basis_paths": {
                "contrast": basis,
                "pca_full_update": basis,
                "random_full_update": basis,
            },
            "implicit_hash_lift": {"projection_dim": 512, "projection_seed": 20260820},
            "selected": {
                "weak": {"rank": 1, "alpha": .25},
                "strong": {"rank": 5, "alpha": 1.0},
                "head_ablation": {"rank": 5, "alpha": 1.0},
                "single_full_update_basis": {"rank": 5, "alpha": 1.0},
                "clip_gaussian_weak": {"clip_norm": .1, "noise_l2_ratio": .05},
                "clip_gaussian_strong": {"clip_norm": .08, "noise_l2_ratio": .15},
            },
            "l2_budget": {
                "target_data_used": False,
                "weak_max_relative_update_distortion_l2": .02,
                "strong_max_relative_update_distortion_l2": .14,
            },
        }
        head = {
            "status": "LOCKED",
            "provenance": {"target_data_used_for_basis_or_tuning": False},
            "basis": {"path": basis},
            "selected": {"rank": 8, "alpha": 1.0},
        }
        controls = _defense_configs(base_config, matched)
        complete = _complete_configs(base_config, matched, head)
        self.assertEqual(len(controls), 70)
        self.assertEqual(len(complete), 8)
        for name, cfg in controls:
            defense = cfg["defense"]
            if defense["name"] == "streaming_clip_gaussian":
                self.assertNotIn("max_relative_update_distortion_l2", defense)
            else:
                rho = .02 if "weak" in name else .14
                self.assertAlmostEqual(defense["max_relative_update_distortion_l2"], rho)
        for _, cfg in complete:
            defense = cfg["defense"]
            self.assertAlmostEqual(defense["max_relative_update_distortion_l2"], .14)
            self.assertEqual(len(defense["layers"]), 5)
            self.assertEqual(defense["layers"][0]["group"], "classifier_head")
            self.assertEqual(defense["layers"][0]["rank"], 8)
            self.assertEqual(defense["layers"][0]["alpha"], 1.0)
            self.assertEqual({x["group"] for x in defense["layers"]},
                             {"classifier_head", "fusion", "missing_modality", "image_encoder", "text_encoder"})

    def test_numerical_verifier_is_always_strict(self):
        src = (ROOT / "scripts/acm_revision/verify_paper_table7.py").read_text()
        self.assertIn("EXPECTED_TABLE7", src)
        self.assertIn("if candidate_count != 54:", src)
        self.assertIn("metadata.get(\"paper_milestone_rounds\")", src)
        self.assertNotIn('parser.add_argument("--strict-protocol"', src)

    def test_traceability(self):
        text = (ROOT / "docs/paper_repro/TRACEABILITY.csv").read_text()
        for name in ("Table 1", "Table 2", "Table 3", "Table 4", "Table 5",
                     "Table 6", "Table 7", "Table 8", "Table D1", "Table E3", "Table F1"):
            self.assertIn(name, text)

    def test_actual_dispatch_budget_and_missing_budget_rejected(self):
        try:
            import torch
            import numpy as np
            import tempfile
            from src.acm_revision.defense_dispatch import apply_defense_to_state
        except ImportError:
            self.skipTest("Full Torch/Numpy not installed; run in l2-runtime CI")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "head.npz"
            np.savez_compressed(path, classifier_head=np.array([[1.0], [0.0], [0.0]], dtype=np.float32))
            before = {"fusion.classifier.weight": torch.tensor([0.0, 0.0, 0.0])}
            trained = {"fusion.classifier.weight": torch.tensor([2.0, 1.0, 0.0])}
            cfg = {
                "name": "contrast_filter", "group": "classifier_head",
                "basis_path": str(path), "rank": 1, "alpha": 1.0,
                "max_relative_update_distortion_l2": .1,
            }
            defended, log = apply_defense_to_state(before, trained, cfg, seed=0)
            self.assertTrue(log["l2_budget"]["budget_verified"])
            raw = trained["fusion.classifier.weight"] - before["fusion.classifier.weight"]
            transmitted = defended["fusion.classifier.weight"] - before["fusion.classifier.weight"]
            measured = torch.linalg.vector_norm(transmitted - raw).item()
            bound = .1 * torch.linalg.vector_norm(raw).item()
            self.assertLessEqual(measured, bound + 1e-6)
            self.assertTrue(log["l2_budget"]["is_active"])
            del cfg["max_relative_update_distortion_l2"]
            with self.assertRaises(ValueError):
                apply_defense_to_state(before, trained, cfg, seed=0)

    def test_true_original_coordinate_l2_cap_runtime(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU Torch not installed; validated in separate runtime CI job")
        before = {"a": torch.tensor([1., 2., 3.]), "b": torch.tensor([-1., 5.])}
        trained = {"a": torch.tensor([3., 2., 3.]), "b": torch.tensor([-1., 7.])}
        filtered = {"a": torch.tensor([0., 0., 0.]), "b": torch.tensor([0., 0.])}
        report = budget.apply_common_l2_cap_inplace(before, trained, filtered, .2, chunk_size=2)
        self.assertTrue(report["budget_verified"])
        self.assertTrue(report["is_active"])
        self.assertLessEqual(report["applied_change_l2"], .2 * report["raw_update_l2"] + 1e-6)
        # All original coordinates included, not a reduced projection.
        self.assertEqual(report["num_original_floating_coordinates"], 5)
        self.assertGreater(report["raw_update_l2"], 0)
        # Distortion should not be forced equal to other algorithms: only capped.
        no_change = {k: (trained[k] - before[k]).clone() for k in before}
        untouched = budget.apply_common_l2_cap_inplace(before, trained, no_change, .2)
        self.assertFalse(untouched["is_active"])
        self.assertAlmostEqual(untouched["applied_change_l2"], 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
