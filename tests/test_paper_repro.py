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
l2ref = from_path("paper_l2_reference", "src/acm_revision/l2_reference.py")


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

    def test_rounded_l2_reference_thresholds(self):
        expected = {"strong": {"classifier_head": 1, "full_update": 1},
                    "weak": {"classifier_head": 1, "full_update": 1}}
        self.assertEqual(l2ref.MEAN_RELATIVE_L2_THRESHOLDS, expected)
        report = l2ref.audit_existing_means()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["rows"]), 12)
        self.assertTrue(report["no_training_updates_modified"])
        self.assertFalse(report["historical_hard_cap_verified"])
        self.assertFalse(report["full_update_original_space_L2_verified"])

    def test_training_update_l2_budget_is_enforced(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("Runtime update-level clipping test requires torch (GPU/FL environment)")
        import torch
        from src.acm_revision.defense_dispatch import _enforce_relative_l2_budget

        before = {
            "fusion.classifier.weight": torch.zeros(2),
            "clip.vision_model.example": torch.zeros(2),
        }
        after = {
            "fusion.classifier.weight": torch.tensor([1.0, 0.0]),
            "clip.vision_model.example": torch.tensor([1.0, 0.0]),
        }
        delta = {
            "fusion.classifier.weight": torch.tensor([3.0, 0.0]),
            "clip.vision_model.example": torch.tensor([5.0, 0.0]),
        }
        detail = _enforce_relative_l2_budget(
            before, after, delta, {"classifier_head": 1.0, "full_update": 1.0}
        )
        self.assertTrue(detail["activated"])
        self.assertLess(detail["perturbation_scale"], 1.0)
        for group, limit in detail["limits"].items():
            self.assertLessEqual(detail["relative_after"][group], limit + 1e-6)
        self.assertAlmostEqual(
            float(delta["fusion.classifier.weight"][0]),
            1.0 + 2.0 * detail["perturbation_scale"], places=5,
        )
        self.assertAlmostEqual(
            float(delta["clip.vision_model.example"][0]),
            1.0 + 4.0 * detail["perturbation_scale"], places=5,
        )

        already_compliant = {
            "fusion.classifier.weight": torch.tensor([1.25, 0.0]),
            "clip.vision_model.example": torch.tensor([1.5, 0.0]),
        }
        detail = _enforce_relative_l2_budget(
            before, after, already_compliant,
            {"classifier_head": 1.0, "full_update": 1.0}
        )
        self.assertFalse(detail["activated"])
        self.assertEqual(detail["perturbation_scale"], 1.0)
        self.assertEqual(float(already_compliant["fusion.classifier.weight"][0]), 1.25)
        self.assertEqual(float(already_compliant["clip.vision_model.example"][0]), 1.5)

    def test_common_budget_is_wired_to_filtered_methods(self):
        dispatcher = (ROOT / "src/acm_revision/defense_dispatch.py").read_text()
        self.assertIn("_enforce_relative_l2_budget", dispatcher)
        self.assertIn("return _finalize_filter(before_state, after_state, delta, defense_cfg", dispatcher)
        from src.acm_revision.l2_reference import MEAN_RELATIVE_L2_THRESHOLDS
        self.assertEqual(
            MEAN_RELATIVE_L2_THRESHOLDS["weak"],
            MEAN_RELATIVE_L2_THRESHOLDS["strong"],
        )

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

    def test_70_and_8_generated_formal_configs_preserve_defense_parameters(self):
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
        for _, cfg in controls:
            self.assertNotIn("max_relative_update_distortion_l2", cfg["defense"])
            if cfg["defense"]["name"] == "streaming_clip_gaussian":
                self.assertNotIn("relative_l2_budget", cfg["defense"])
            else:
                self.assertEqual(cfg["defense"]["relative_l2_budget"],
                                 {"classifier_head": 1.0, "full_update": 1.0})
        for _, cfg in complete:
            defense = cfg["defense"]
            self.assertNotIn("max_relative_update_distortion_l2", defense)
            self.assertEqual(defense["relative_l2_budget"],
                             {"classifier_head": 1.0, "full_update": 1.0})
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



if __name__ == "__main__":
    unittest.main()
