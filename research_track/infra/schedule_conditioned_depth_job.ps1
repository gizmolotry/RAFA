param(
  [Parameter(Mandatory = $true)][string]$JobJson,
  [Parameter(Mandatory = $true)][string]$StatusJson,
  [string]$TaskName = "",
  [int]$DelaySeconds = 30
)

$ErrorActionPreference = 'Stop'

function Quote-Arg {
  param([string]$Value)
  return '"' + ($Value -replace '"', '""') + '"'
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$job = Get-Content $JobJson -Raw | ConvertFrom-Json
$jobName = [string]$job.name

if ([string]::IsNullOrWhiteSpace($TaskName)) {
  $TaskName = "RAFA_" + $jobName
}

$launcher = Join-Path $repoRoot 'research_track\infra\launch_conditioned_depth_job.ps1'
$schedulerMeta = Join-Path $repoRoot ("research_track\infra\summaries\{0}.scheduler.json" -f $jobName)
$wrapperDir = Join-Path $repoRoot 'tmp\scheduled_tasks'
$wrapperPath = Join-Path $wrapperDir ($TaskName + '.cmd')

$start = (Get-Date).AddSeconds([Math]::Max(15, $DelaySeconds))
$st = $start.ToString('HH:mm')
$sd = $start.ToString('MM/dd/yyyy')
$statusDir = Split-Path -Parent $StatusJson
if ($statusDir) {
  New-Item -ItemType Directory -Force $statusDir | Out-Null
}

$psArgs = @(
  '-NoProfile',
  '-ExecutionPolicy', 'Bypass',
  '-File', $launcher,
  '-JobJson', (Resolve-Path $JobJson).Path,
  '-StatusJson', $StatusJson
)

$psExe = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
$wrapperCmd = '@echo off' + "`r`n" + (Quote-Arg $psExe) + ' ' + (($psArgs | ForEach-Object { Quote-Arg $_ }) -join ' ') + "`r`n"
New-Item -ItemType Directory -Force $wrapperDir | Out-Null
Set-Content -Path $wrapperPath -Value $wrapperCmd -Encoding ASCII

$taskRun = 'cmd.exe /c ' + (Quote-Arg $wrapperPath)

try {
  schtasks /Delete /TN $TaskName /F | Out-Null
} catch {
}

$createOutput = schtasks /Create /SC ONCE /TN $TaskName /TR $taskRun /ST $st /SD $sd /F
$runOutput = schtasks /Run /TN $TaskName

$payload = [ordered]@{
  task_name = $TaskName
  job_name = $jobName
  job_json = (Resolve-Path $JobJson).Path
  status_json = $StatusJson
  launcher = $launcher
  wrapper_cmd = $wrapperPath
  scheduler_run_command = $taskRun
  start_time_local = $start.ToString('o')
  created_utc = [DateTime]::UtcNow.ToString('o')
  create_output = ($createOutput | Out-String).Trim()
  run_output = ($runOutput | Out-String).Trim()
}

New-Item -ItemType Directory -Force (Split-Path -Parent $schedulerMeta) | Out-Null
($payload | ConvertTo-Json -Depth 8) | Set-Content -Path $schedulerMeta -Encoding UTF8
($payload | ConvertTo-Json -Depth 8)
