# Input-geometry and detection-budget ablations. Appends each command to results/command_log.txt.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $python = '.\.venv\Scripts\python.exe'
    $log = 'results/command_log.txt'
    function Invoke-Logged([string[]]$arguments) {
        Add-Content -Encoding utf8 $log ("[{0:u}] python {1}" -f (Get-Date).ToUniversalTime(), ($arguments -join ' '))
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Failed: $($arguments -join ' ')" }
    }
    $runs = @(
        @{model = 'rtdetr_l'; dataset = 'kitti'; extra = @('--pad-square'); tag = '_padsq'},
        @{model = 'yolo11s'; dataset = 'kitti'; extra = @('--imgsz', '1280'); tag = '_imgsz1280'},
        @{model = 'yolo11n'; dataset = 'kitti'; extra = @('--imgsz', '1280'); tag = '_imgsz1280'},
        @{model = 'yolo11s'; dataset = 'coco'; extra = @('--max-det', '300'); tag = '_maxdet300'}
    )
    foreach ($r in $runs) {
        Invoke-Logged (@('scripts/run_inference.py', '--model', $r.model, '--dataset', $r.dataset) + $r.extra)
        Invoke-Logged @('scripts/evaluate.py', '--model', $r.model, '--dataset', $r.dataset,
                        '--predictions', "predictions/$($r.dataset)/$($r.model)$($r.tag).json")
    }
} finally {
    Pop-Location
}
