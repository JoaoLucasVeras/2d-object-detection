# Run every model on every dataset, then score them. Appends each command to results/command_log.txt.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $python = '.\.venv\Scripts\python.exe'
    $models = (Get-Content configs/models.json -Raw | ConvertFrom-Json).models.id
    $log = 'results/command_log.txt'
    function Invoke-Logged([string[]]$arguments) {
        Add-Content -Encoding utf8 $log ("[{0:u}] python {1}" -f (Get-Date).ToUniversalTime(), ($arguments -join ' '))
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Failed: $($arguments -join ' ')" }
    }
    Invoke-Logged @('scripts/prepare_data.py')
    foreach ($dataset in 'coco', 'kitti') {
        foreach ($model in $models) {
            Invoke-Logged @('scripts/run_inference.py', '--model', $model, '--dataset', $dataset)
            Invoke-Logged @('scripts/evaluate.py', '--model', $model, '--dataset', $dataset)
        }
    }
} finally {
    Pop-Location
}
