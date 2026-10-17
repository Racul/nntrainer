# SPDX-License-Identifier: Apache-2.0
# Usage: .\eval\h2o\windows_h2o_eval.ps1 [MODEL_DIR] [--check] [options]
$ErrorActionPreference = "Stop"
$pythonExe = $env:PYTHON
$pythonArgs = @()
if (-not $pythonExe) {
    $venvPython = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $pythonExe = $venvPython
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonExe = "py"
        $pythonArgs = @("-3")
    } else {
        $pythonExe = "python"
    }
}
& $pythonExe @pythonArgs (Join-Path $PSScriptRoot "h2o_eval.py") @args
exit $LASTEXITCODE
