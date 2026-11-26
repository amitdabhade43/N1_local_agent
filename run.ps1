$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}
$python = ".\.venv\Scripts\python.exe"

# Install only when the pinned dependency set changed, so a normal launch needs no network
# and never upgrades pip or packages behind your back.
$lock = if (Test-Path "requirements.lock") { "requirements.lock" } else { "requirements.txt" }
$hash = (Get-FileHash $lock -Algorithm SHA256).Hash
$stamp = ".venv\.deps-hash"
if (-not (Test-Path $stamp) -or ((Get-Content $stamp -Raw).Trim() -ne $hash)) {
    Write-Host "Installing dependencies from $lock ..."
    & $python -m pip install -r $lock
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed (see output above)." }
    Set-Content -Path $stamp -Value $hash -NoNewline   # written only after a successful install
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env. Review model IDs and endpoints if needed."
}

# app.py loads .env itself (python-dotenv) and defaults HOST=127.0.0.1, PORT=8080.
Write-Host "Starting NinjaOne Local Agent (see HOST/PORT in .env; default http://127.0.0.1:8080)"
& $python app.py
