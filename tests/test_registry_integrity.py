from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "registries"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class RegistryIntegrityTests(unittest.TestCase):
    def test_all_registry_files_exist(self) -> None:
        for name in [
            "RUNTIME_REGISTRY.json",
            "CHECKPOINT_REGISTRY.json",
            "LINEAGE_REGISTRY.json",
            "SHARED_MODULE_REGISTRY.json",
            "PIPELINE_DAG.json",
            "WORKSTREAM_REGISTRY.json",
        ]:
            self.assertTrue((REGISTRY_DIR / name).exists(), name)

    def test_runtime_registry_paths_exist(self) -> None:
        data = load_json(REGISTRY_DIR / "RUNTIME_REGISTRY.json")
        for row in data["runtimes"]:
            self.assertTrue((ROOT / row["manifest"]).exists(), row["manifest"])
            self.assertTrue((ROOT / row["entrypoint"]).exists(), row["entrypoint"])

    def test_checkpoint_registry_paths_exist(self) -> None:
        data = load_json(REGISTRY_DIR / "CHECKPOINT_REGISTRY.json")
        runtime_ids = {row["id"] for row in load_json(REGISTRY_DIR / "RUNTIME_REGISTRY.json")["runtimes"]}
        schema_ids = {row["id"] for row in data["schema_families"]}
        for row in data["tracked_checkpoints"]:
            self.assertTrue((ROOT / row["path"]).exists(), row["path"])
            self.assertIn(row["runtime"], runtime_ids)
            self.assertIn(row["schema"], schema_ids)
            self.assertTrue((ROOT / row["canonical_entrypoint"]).exists(), row["canonical_entrypoint"])

    def test_lineage_registry_paths_exist(self) -> None:
        data = load_json(REGISTRY_DIR / "LINEAGE_REGISTRY.json")
        runtime_ids = {row["id"] for row in load_json(REGISTRY_DIR / "RUNTIME_REGISTRY.json")["runtimes"]}
        for row in data["lineages"]:
            self.assertTrue((ROOT / row["path"]).exists(), row["path"])
            self.assertTrue((ROOT / row["constitution"]).exists(), row["constitution"])
            self.assertIn(row["primary_runtime"], runtime_ids)
            for runtime_id in row.get("secondary_runtimes", []):
                self.assertIn(runtime_id, runtime_ids)

    def test_shared_modules_exist(self) -> None:
        data = load_json(REGISTRY_DIR / "SHARED_MODULE_REGISTRY.json")
        runtime_ids = {row["id"] for row in load_json(REGISTRY_DIR / "RUNTIME_REGISTRY.json")["runtimes"]}
        for row in data["modules"]:
            self.assertTrue((ROOT / row["path"]).exists(), row["path"])
            for runtime_id in row["used_by"]:
                self.assertIn(runtime_id, runtime_ids)

    def test_pipeline_dag_is_valid_and_acyclic(self) -> None:
        dag = load_json(REGISTRY_DIR / "PIPELINE_DAG.json")
        nodes = dag["nodes"]
        edges = dag["edges"]
        node_ids = [row["id"] for row in nodes]
        self.assertEqual(len(node_ids), len(set(node_ids)))
        for row in nodes:
            self.assertTrue((ROOT / row["path"]).exists(), row["path"])

        adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
        for row in edges:
            self.assertIn(row["from"], adjacency)
            self.assertIn(row["to"], adjacency)
            adjacency[row["from"]].append(row["to"])
            indegree[row["to"]] += 1

        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        seen: list[str] = []
        while queue:
            node_id = queue.pop(0)
            seen.append(node_id)
            for nxt in adjacency[node_id]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        self.assertEqual(len(seen), len(node_ids), "PIPELINE_DAG.json contains a cycle")

    def test_workstream_registry_is_valid(self) -> None:
        data = load_json(REGISTRY_DIR / "WORKSTREAM_REGISTRY.json")
        runtime_ids = {row["id"] for row in load_json(REGISTRY_DIR / "RUNTIME_REGISTRY.json")["runtimes"]}
        lineage_ids = {row["id"] for row in load_json(REGISTRY_DIR / "LINEAGE_REGISTRY.json")["lineages"]}

        import subprocess

        branch_output = subprocess.check_output(
            ["git", "branch", "--list"],
            cwd=ROOT,
            text=True,
        )
        branches = {line.strip().lstrip("*+ ").strip() for line in branch_output.splitlines() if line.strip()}

        for row in data["workstreams"]:
            self.assertIn(row["branch"], branches)
            primary_runtime = row.get("primary_runtime")
            primary_lineage = row.get("primary_lineage")
            if primary_runtime is not None:
                self.assertIn(primary_runtime, runtime_ids)
            if primary_lineage is not None:
                self.assertIn(primary_lineage, lineage_ids)
            for path in row.get("owned_paths", []):
                self.assertTrue((ROOT / path).exists(), path)
            for path in row.get("reference_only_paths", []):
                self.assertTrue((ROOT / path).exists(), path)


if __name__ == "__main__":
    unittest.main()
