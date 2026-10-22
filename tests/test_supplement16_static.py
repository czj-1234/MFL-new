"""Offline, dataset-free source/plan guards for the 16-run supplement."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Supplement16StaticTest(unittest.TestCase):
    def setUp(self):
        p = ROOT / "src/acm_revision/supplement16.py"
        self.text = p.read_text(encoding="utf-8")
        self.tree = ast.parse(self.text)

    def _constant(self, name):
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                    return ast.literal_eval(node.value)
        raise AssertionError(f"Missing {name}")

    def test_exact_matrix(self):
        expected = self._constant("EXPECTED")
        self.assertEqual(len(expected), 8)
        self.assertEqual([seed for pop, seed, _ in expected if pop == "target"],
                         [42, 43, 44, 45, 46])
        self.assertEqual([seed for pop, seed, _ in expected if pop == "shadow_train"],
                         [142, 143])
        self.assertEqual([seed for pop, seed, _ in expected if pop == "shadow_val"], [242])
        self.assertEqual(self._constant("MILESTONES"),
                         (1, 10, 30, 50, 75, 100, 125, 150))
        self.assertEqual(self._constant("OTHER_GROUPS"),
                         ("fusion", "missing_modality", "image_encoder", "text_encoder"))

    def test_missing_reference_yaml_is_rebuilt_without_refitting(self):
        helper = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "ensure_original_reference_config")
        src = ast.get_source_segment(self.text, helper)
        self.assertIn('original_prep.make_reference_config(', src)
        self.assertIn('output_root=reference.parent', src)
        self.assertIn('actual != expected', src)
        self.assertIn('candidate.replace(REF_CONFIG)', src)
        self.assertIn('if REF_CONFIG.is_file()', src)
        self.assertNotIn('local_train(', src)
        self.assertNotIn('fit_shadow_only_preparation(', src)
        main = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "prepare_uncentered")
        psrc = ast.get_source_segment(self.text, main)
        self.assertLess(
            psrc.index('checkpoints = [require_file('),
            psrc.index('ensure_original_reference_config(reference)'),
        )
        self.assertLess(
            psrc.index('ensure_original_reference_config(reference)'),
            psrc.index('source_hashes["reference_config"]'),
        )

    def test_no_target_data_in_basis_preparation(self):
        func = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "prepare_uncentered")
        source = ast.get_source_segment(self.text, func)
        self.assertIn('shadow_train.json', source)
        self.assertIn('target_data_used', self.text)
        self.assertIn('min_cosine < .995', source)
        self.assertNotIn('target.json', source)

    def test_no_fallback_to_changed_head(self):
        self.assertIn('Attacker-refined head must not change', self.text)
        self.assertIn('d["layers"][0] == frozen["layers"][0]', self.text)
        self.assertIn('full_group_projection_dim"] == 2048', self.text)
        self.assertIn('prep._fit_lowrank(rows[group], prep.MAX_RANK)', self.text)

    def test_per_update_original_parameter_distortion(self):
        src = (ROOT / "src/acm_revision/defense_dispatch.py").read_text(encoding="utf-8")
        capture = (ROOT / "src/acm_revision/fl_resume.py").read_text(encoding="utf-8")
        for key in ("raw_update_l2", "actual_difference_l2",
                    '"norm_space": "exact_original_model_update"'):
            self.assertIn(key, src)
        for key in ("defense_l2_head_difference_norm", "defense_l2_full_difference_norm",
                    "defense_l2_head_post", "defense_l2_full_post"):
            self.assertIn(key, capture)

    def test_relative_l2_budget_scale_never_calls_single_arg_min_on_float(self):
        # Evaluate the exact scalar scale expression with stdlib-only fixtures,
        # even on GitHub CI machines without torch or private data.
        source = (ROOT / "src/acm_revision/defense_dispatch.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        func = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_enforce_relative_l2_budget")
        scale_stmt = next(n for n in ast.walk(func) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "scale" for t in n.targets))
        expression = ast.fix_missing_locations(ast.Expression(body=scale_stmt.value))
        compiled = compile(expression, "<l2-budget-scale>", "eval")
        limits = {"classifier_head": 1.0, "full_update": 1.0}
        for head, full, expected in (
            (0.0, 0.0, 1.0),
            (0.2, 0.9, 1.0),
            (1.0, 1.0, 1.0),
            (2.0, 0.5, 0.5),
            (0.25, 4.0, 0.25),
            (2.0, 4.0, 0.25),
        ):
            namespace = {
                "min": min,
                "limits": limits,
                "relative_before": {"classifier_head": head, "full_update": full},
                "expected": set(limits),
            }
            actual = eval(compiled, namespace)
            self.assertAlmostEqual(actual, expected, places=10)

    def test_foreground_live_progress_and_failure_propagation(self):
        job_runner = (ROOT / "scripts/acm_revision/run_defense70_job.sh").read_text(encoding="utf-8")
        trainer = (ROOT / "src/acm_revision/fl_resume.py").read_text(encoding="utf-8")
        self.assertIn('DEFENSE70_LIVE_STDOUT:-0', job_runner)
        self.assertIn('2>&1 | tee "${LOG}" &', job_runner)
        self.assertIn('set -uo pipefail', job_runner)
        self.assertIn('MFL_LIVE_PROGRESS', trainer)
        self.assertIn('[FL PROGRESS] round=', trainer)
        for side, script in (
            ("A", "serverA_uncentered_8.sh"),
            ("B", "serverB_c09_8.sh"),
        ):
            text = (ROOT / "scripts/acm_revision" / script).read_text(encoding="utf-8")
            self.assertIn('export DEFENSE70_LIVE_STDOUT=1', text)
            self.assertIn('export MFL_LIVE_PROGRESS=1', text)
            self.assertIn('run_queue "$GPU0" 0 2>&1 | awk', text)
            self.assertIn('run_queue "$GPU1" 1 2>&1 | awk', text)
            self.assertIn('tee -a "$DEFENSE70_LOG_ROOT/gpu0.queue.log"', text)
            self.assertIn('tee -a "$DEFENSE70_LOG_ROOT/gpu1.queue.log"', text)
            self.assertIn('flock -n "$SERVER_LOCK_FD"', text)
            wrapper = (ROOT / f"start_server{side}_live.sh").read_text(encoding="utf-8")
            self.assertIn(f'exec bash scripts/acm_revision/{script}', wrapper)

    def test_gpu_start_gate_is_7500_mib_on_both_servers(self):
        for name in ("serverA_uncentered_8.sh", "serverB_c09_8.sh"):
            script = (ROOT / "scripts/acm_revision" / name).read_text(encoding="utf-8")
            self.assertIn('DEFENSE70_MIN_FREE_MIB="${MIN_FREE_MIB:-7500}"', script)
        runner = (ROOT / "scripts/acm_revision/run_defense70_job.sh").read_text(encoding="utf-8")
        self.assertIn('(( free_mib >= MIN_FREE_MIB ))', runner)
        self.assertIn('MIN_FREE_MIB="${DEFENSE70_MIN_FREE_MIB:-23000}"', runner)

    def test_both_server_launchers_exist(self):
        for phase, name in (("uncentered", "serverA_uncentered_8.sh"),
                            ("c09", "serverB_c09_8.sh")):
            text = (ROOT / "scripts/acm_revision" / name).read_text(encoding="utf-8")
            self.assertIn(f"--phase {phase}", text)
            self.assertIn('run_queue "$GPU0" 0', text)
            self.assertIn('run_queue "$GPU1" 1', text)
            self.assertIn('run_defense70_job.sh', text)
            self.assertIn('verify --phase "$phase"', text)


if __name__ == "__main__":
    unittest.main()
