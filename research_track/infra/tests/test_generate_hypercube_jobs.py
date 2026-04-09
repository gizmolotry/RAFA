from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from research_track.infra.generate_hypercube_jobs import generate_jobs


class HypercubeJobGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        manifest_path = repo_root / "research_track" / "infra" / "hypercube_manifest.yaml"
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)

    def test_generates_expected_job_count(self) -> None:
        jobs = generate_jobs(self.manifest)
        # 2^5 rafa combinations + 5 named runs.
        self.assertEqual(len(jobs), 37)

    def test_job_names_are_unique(self) -> None:
        jobs = generate_jobs(self.manifest)
        names = [j["name"] for j in jobs]
        self.assertEqual(len(names), len(set(names)))

    def test_contains_baseline_anchor(self) -> None:
        jobs = generate_jobs(self.manifest)
        names = {j["name"] for j in jobs}
        self.assertIn("baseline_v3_anchor", names)

    def test_contains_prompt_conditioned_smoke(self) -> None:
        jobs = generate_jobs(self.manifest)
        by_name = {j["name"]: j for j in jobs}
        self.assertIn("prompt_conditioned_clap_smoke", by_name)
        self.assertTrue(by_name["prompt_conditioned_clap_smoke"]["text_prompt_conditioned"])

    def test_contains_ifs_control_smoke(self) -> None:
        jobs = generate_jobs(self.manifest)
        by_name = {j["name"]: j for j in jobs}
        self.assertIn("prompt_conditioned_ifs_control_smoke", by_name)
        self.assertEqual(by_name["prompt_conditioned_ifs_control_smoke"]["text_condition_mode"], "ifs_control")
        self.assertFalse(by_name["prompt_conditioned_ifs_control_smoke"]["model_text_conditioned"])

    def test_contains_hybrid_smoke(self) -> None:
        jobs = generate_jobs(self.manifest)
        by_name = {j["name"]: j for j in jobs}
        self.assertIn("prompt_conditioned_hybrid_smoke", by_name)
        self.assertEqual(by_name["prompt_conditioned_hybrid_smoke"]["text_condition_mode"], "hybrid")
        self.assertTrue(by_name["prompt_conditioned_hybrid_smoke"]["model_text_conditioned"])

    def test_contains_semantic_tension_smoke(self) -> None:
        jobs = generate_jobs(self.manifest)
        by_name = {j["name"]: j for j in jobs}
        self.assertIn("semantic_tension_steering_smoke", by_name)
        self.assertEqual(by_name["semantic_tension_steering_smoke"]["semantic_condition_mode"], "tension_envelope")
        self.assertFalse(by_name["semantic_tension_steering_smoke"]["text_prompt_conditioned"])
        self.assertEqual(by_name["semantic_tension_steering_smoke"]["phase_qset"], [2, 3, 4, 5, 6, 7, 8, 11, 12, 13])


if __name__ == "__main__":
    unittest.main()
