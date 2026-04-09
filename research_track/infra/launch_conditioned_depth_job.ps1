param(
  [Parameter(Mandatory = $true)][string]$JobJson,
  [Parameter(Mandatory = $true)][string]$StatusJson
)

$ErrorActionPreference = 'Stop'

function Write-Status {
  param(
    [string]$Name,
    [string]$Phase,
    [string]$Status,
    [string]$UpdatedUtc,
    [string]$JobPath,
    [string]$ErrorMessage = "",
    [string]$LaunchLog = "",
    [string]$LaunchErr = ""
  )
  $payload = [ordered]@{
    name = $Name
    phase = $Phase
    status = $Status
    updated_utc = $UpdatedUtc
    job_json = $JobPath
    error = $ErrorMessage
    launch_log = $LaunchLog
    launch_err = $LaunchErr
  }
  $dir = Split-Path -Parent $StatusJson
  if ($dir) {
    New-Item -ItemType Directory -Force $dir | Out-Null
  }
  ($payload | ConvertTo-Json -Depth 8) | Set-Content -Path $StatusJson -Encoding UTF8
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$job = Get-Content $JobJson -Raw | ConvertFrom-Json
$jobName = [string]$job.name
$pythonExe = ""
if ($job.PSObject.Properties.Name -contains 'python_executable') {
  $pythonExe = [string]$job.python_executable
}
if ([string]::IsNullOrWhiteSpace($pythonExe)) {
  $pythonExe = [Environment]::GetEnvironmentVariable('RAFA_PYTHON_EXE')
}
if ([string]::IsNullOrWhiteSpace($pythonExe)) {
  $pythonExe = 'C:\Users\Andrew\miniconda3\envs\RAFA\python.exe'
}
$logsDir = Join-Path $repoRoot 'logs\conditioned_depth'
New-Item -ItemType Directory -Force $logsDir | Out-Null
$launchStamp = "{0}_{1}" -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'), $PID
$launchLog = Join-Path $logsDir ("pslaunch_{0}_{1}.log" -f $jobName, $launchStamp)
$launchErr = Join-Path $logsDir ("pslaunch_{0}_{1}.err.log" -f $jobName, $launchStamp)

Write-Status -Name $jobName -Phase 'bootstrap' -Status 'running' -UpdatedUtc ([DateTime]::UtcNow.ToString('o')) -JobPath $JobJson -LaunchLog $launchLog -LaunchErr $launchErr

try {
  & $pythonExe -u (Join-Path $repoRoot 'research_track\infra\run_conditioned_depth_job.py') --job-json $JobJson --status-json $StatusJson 1>> $launchLog 2>> $launchErr
  if ($LASTEXITCODE -ne 0) {
    throw "run_conditioned_depth_job.py exited with code $LASTEXITCODE"
  }
} catch {
  Write-Status -Name $jobName -Phase 'error' -Status 'error' -UpdatedUtc ([DateTime]::UtcNow.ToString('o')) -JobPath $JobJson -ErrorMessage $_.Exception.Message -LaunchLog $launchLog -LaunchErr $launchErr
  throw
}
