$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PATH = [Environment]::GetEnvironmentVariable("PATH", "User") + ";" +
    [Environment]::GetEnvironmentVariable("PATH", "Machine")

Write-Host "Running unit tests..." -ForegroundColor Cyan
uv run pytest tests/customer_support -q

$EnvLines = Get-Content -LiteralPath ".env" -ErrorAction SilentlyContinue
$HasDashScopeKey = [bool]($EnvLines | Where-Object {
    $_ -match "^DASHSCOPE_API_KEY=.+"
})
$ForcesMock = [bool]($EnvLines | Where-Object {
    $_ -match "^CUSTOMER_SUPPORT_MODE=mock\s*$"
})

Write-Host ""
if ($HasDashScopeKey -and -not $ForcesMock) {
    Write-Host "Online Qwen mode detected." -ForegroundColor Yellow
    Write-Host "Skipping the 9-case HTTP batch to avoid unintended API charges."
    Write-Host "Use the three manual UI checks in START_HERE.md instead."
}
else {
    Write-Host "Running offline HTTP end-to-end checks..." -ForegroundColor Cyan
    uv run python scripts/demo_check.py
}
