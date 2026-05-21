# Branch Bundle Manifests - 2026-05-21

Generated from the dirty source worktree on `codex/circleworld-cleanup-cartography-2026-05-21`.

These files are manifests only. They do not authorize deletion and they do not imply promotion.

Each `lane_*.txt` file is path-only. Each `lane_*_status.tsv` file keeps the original Git status plus path for auditability.

| Lane | Path manifest | Status receipt | Files | Target branch |
| --- | --- | --- | ---: | --- |
| circleworld_runtime_or_spike | lane_circleworld_runtime_or_spike.txt | lane_circleworld_runtime_or_spike_status.tsv | 116 | codex/circleworld |
| circleworld_signature_spike | lane_circleworld_signature_spike.txt | lane_circleworld_signature_spike_status.tsv | 3 | codex/circleworld-signature-spike-2026-05-21 |
| repo_migration_project_cartography | lane_repo_migration_project_cartography.txt | lane_repo_migration_project_cartography_status.tsv | 19 | codex/repo-migration |
| research_track | lane_research_track.txt | lane_research_track_status.tsv | 113 | codex/research-track |
| runtime_contracts | lane_runtime_contracts.txt | lane_runtime_contracts_status.tsv | 16 | codex/runtime-contracts |

Application mechanism: copy listed files from `D:\RAFA` into clean sibling worktrees under `D:\RAFA_worktrees`, preserving paths. This handles tracked modifications and untracked files without staging the dirty source checkout.
