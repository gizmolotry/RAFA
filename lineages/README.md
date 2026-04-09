# Lineages

`lineages/` contains branch-like experiment families without requiring separate git branches.

## Folders
- `01_parent_graduation/`: old maximal hybrid baseline
- `02_local_ablations/`: component knockouts
- `03_subtractive_fork/`: subtractive purity / stage-4 fork
- `04_positive_replacement/`: Circleworld / Hardy-Littlewood positive replacement lane

## How to run
Use [launch_lineage.ps1](../launch_lineage.ps1):

```powershell
.\launch_lineage.ps1 04 ablate_formalization.py --mode active_packets --depth 2
```

If you are testing a conceptual architectural branch instead of the current runtime, start here.
