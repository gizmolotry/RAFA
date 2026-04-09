from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from research_track.infra.generate_conditioned_depth_jobs import generate_jobs


class GenerateConditionedDepthJobsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        manifest_path = repo_root / "research_track" / "infra" / "conditioned_depth_manifest.yaml"
        with manifest_path.open("r", encoding="utf-8") as f:
            cls.manifest = yaml.safe_load(f)

    def test_generates_expected_named_runs(self) -> None:
        jobs = generate_jobs(self.manifest)
        self.assertEqual(len(jobs), 3)
        names = {job["name"] for job in jobs}
        self.assertIn("conditioned_depth_prompt_anchor20_ns6_a0_to120", names)
        self.assertIn("conditioned_depth_prompt_real_quick_ns6_a0_to120", names)
        self.assertIn("conditioned_depth_prompt_real_quick_ns8_a1light_to120", names)

    def test_logic_diag_defaults_propagate(self) -> None:
        jobs = generate_jobs(self.manifest)
        first = jobs[0]
        logic = dict(first["logic_diag"])
        self.assertTrue(logic["enabled"])
        self.assertEqual(logic["iterations"], 25)
        self.assertEqual(len(logic["tasks"]), 12)

    def test_anchor_values_are_typed(self) -> None:
        jobs = generate_jobs(self.manifest)
        by_name = {job["name"]: job for job in jobs}
        self.assertFalse(by_name["conditioned_depth_prompt_anchor20_ns6_a0_to120"]["anchor_enabled"])
        self.assertTrue(by_name["conditioned_depth_prompt_real_quick_ns8_a1light_to120"]["anchor_enabled"])
        self.assertEqual(by_name["conditioned_depth_prompt_real_quick_ns8_a1light_to120"]["num_steps"], 8)


if __name__ == "__main__":
    unittest.main()
