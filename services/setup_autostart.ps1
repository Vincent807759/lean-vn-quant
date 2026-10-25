# ==============================================================================
# LEAN VN-QUANT: WINDOWS TASK SCHEDULER AUTO-START CONFIGURATION
# ==============================================================================

$taskName = "VNQuant_AutoPaperTrade"
$servicesDir = $PSScriptRoot
if (-not $servicesDir) { $servicesDir = (Get-Location).Path }
$workingDir = Split-Path -Parent $servicesDir
if (-not $workingDir -or -not (Test-Path (Join-Path $workingDir "services"))) {
    $workingDir = (Get-Location).Path
}

$pythonPath = Join-Path $workingDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonCmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCmd) { $pythonPath = $pythonCmd.Source }
}
$scriptArgument = "services\daily_paper_trader.py"

Write-Host "=================================================================="
Write-Host " REGISTERING SCHEDULED TASK FOR LEAN VN-QUANT DAILY PAPER TRADER"
Write-Host "=================================================================="

# 1. Unregister existing task if present
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[1/3] Removing existing task: $taskName..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
} else {
    Write-Host "[1/3] Initializing new scheduled task..."
}

# 2. Configure Action, Trigger (15:15 Mon-Fri), and Catch-up Settings
Write-Host "[2/3] Configuring Action, Trigger (15:15 Mon-Fri), and StartWhenAvailable..."
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $scriptArgument -WorkingDirectory $workingDir
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 15:15
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

# 3. Register into Windows Task Scheduler
Write-Host "[3/3] Registering task into Windows Task Scheduler..."
$registeredTask = Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Lean VN-Quant: Daily Paper Trader Catch-up and ATO Orders at 15:15"

Write-Host ""
Write-Host "=================================================================="
Write-Host " SCHEDULED TASK SUCCESSFULLY REGISTERED (100% PASS)!"
Write-Host "=================================================================="
Write-Host "  Task Name        : $($registeredTask.TaskName)"
Write-Host "  State            : $($registeredTask.State)"
Write-Host "  Trigger Schedule : 15:15 Mon-Fri"
Write-Host "  Catch-up Offline : StartWhenAvailable = True"
Write-Host "  Execution Command: $pythonPath $scriptArgument"
Write-Host "  Working Directory: $workingDir"
Write-Host "=================================================================="
