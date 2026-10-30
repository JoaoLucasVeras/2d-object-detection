[CmdletBinding()]
param([switch]$SkipSmokeTest)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Install Python 3.11 (64-bit), including the Python launcher, then rerun. See docs/DESKTOP_SETUP.md.'
    }
    & py -3.11 -c "import struct,sys; assert struct.calcsize('P') == 8; print(sys.version)"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 64-bit is required.' }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw 'nvidia-smi was not found. Install/update the NVIDIA driver, restart Windows, then rerun.'
    }
    & nvidia-smi
    if ($LASTEXITCODE -ne 0) { throw 'The NVIDIA driver check failed.' }
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & py -3.11 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
    }
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    & $projectPython -c "import sys; assert sys.version_info[:2] == (3,11), 'Existing .venv must use Python 3.11'"
    if ($LASTEXITCODE -ne 0) { throw 'The existing virtual environment uses the wrong Python version.' }
    & $projectPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip update failed.' }
    & $projectPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & $projectPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
    & $projectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
    New-Item -ItemType Directory -Force -Path results/local, data, checkpoints | Out-Null
    & $projectPython -m pip freeze | Set-Content -Encoding utf8 results/local/environment-freeze.txt
    & $projectPython scripts/check_environment.py --require-cuda
    if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed. Read results/local/environment.json and docs/DESKTOP_SETUP.md.' }
    if (-not $SkipSmokeTest) {
        & $projectPython scripts/smoke_test.py
        if ($LASTEXITCODE -ne 0) { throw 'Model smoke test failed. Read the error above and rerun after fixing it.' }
    }
    Write-Host 'Setup complete. See results/local for environment and smoke-test records.'
    Write-Host 'Next: finalize the two dataset splits and label mapping before full evaluation.'
} finally {
    Pop-Location
}
