# RAFA Lineage Launcher
# Usage: .\launch_lineage.ps1 <lineage_id> <script_name> [args...]
# Lineage IDs: 01 (Parent), 02 (Ablations), 03 (Fork), 04 (Replacement)

param(
    [Parameter(Mandatory=$true)]
    [string]$lineage_id,
    
    [Parameter(Mandatory=$true)]
    [string]$script_name,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$extra_args
)

$lineage_map = @{
    "01" = "01_parent_graduation";
    "02" = "02_local_ablations";
    "03" = "03_subtractive_fork";
    "04" = "04_positive_replacement"
}

$folder = $lineage_map[$lineage_id]
if (-not $folder) {
    Write-Error "Invalid lineage ID. Use 01, 02, 03, or 04."
    exit 1
}

$script_path = "lineages/$folder/$script_name"
if (-not (Test-Path $script_path)) {
    Write-Error "Script not found: $script_path"
    exit 1
}

# AGGRESSIVE QUARANTINE:
# 1. Set PYTHONPATH to include REPO_ROOT/core and REPO_ROOT
$repo_root = Get-Location
$env:PYTHONPATH = "$repo_root;$repo_root\core"

Write-Host "--- RAFA LINEAGE LAUNCHER ---" -ForegroundColor Cyan
Write-Host "Lineage: $folder"
Write-Host "Script:  $script_name"
Write-Host "Root:    $repo_root"
Write-Host "-----------------------------"

# Execute
python $script_path @extra_args
