from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from research_track.infra.generate_logic_diag_jobs import generate_jobs


class LogicDiagJobGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        manifest_path = repo_root / "research_track" / "infra" / "logic_diag_manifest.yaml"
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)

    def test_generates_expected_job_count(self) -> None:
        jobs = generate_jobs(self.manifest)
        self.assertEqual(len(jobs), 24)

    def test_job_names_are_unique(self) -> None:
        jobs = generate_jobs(self.manifest)
        names = [j["name"] for j in jobs]
        self.assertEqual(len(names), len(set(names)))

    def test_contains_bridge_anchor_ic_ns4(self) -> None:
        jobs = generate_jobs(self.manifest)
        names = {j["name"] for j in jobs}
        self.assertIn("logic_diag_bridge_s8000_a1_ic1_ns4", names)

    def test_task_payload_is_present(self) -> None:
        jobs = generate_jobs(self.manifest)
        job = next(j for j in jobs if j["name"] == "logic_diag_prompt_anchor20_a1_ic1_ns4")
        self.assertEqual(len(job["tasks"]), 4)
        self.assertEqual(job["num_steps"] if "num_steps" in job else 4, 4)


if __name__ == "__main__":
    unittest.main()
