$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.13 (64-bit) with tkinter and the Python launcher, then retry.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.\.venv\Scripts\python.exe' scripts/download_models.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed. Demo/ID lookup can still run without models.' }
& '.\.venv\Scripts\python.exe' main.py --check
if ($LASTEXITCODE -ne 0) { throw 'The startup check failed.' }
Write-Host 'Setup complete. Double-click Run STEP Demo.cmd or Run STEP.cmd.'
