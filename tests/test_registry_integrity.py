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


if __name__ == "__main__":
    unittest.main()
