[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$GameRepo = if ($env:GRAVITY_LAB_REPO) { $env:GRAVITY_LAB_REPO } else { Join-Path $Root "gravity-lab" }
$MsysRoot = if ($env:MSYS2_ROOT) { $env:MSYS2_ROOT } else { "C:\msys64" }
$UcrtBin = Join-Path $MsysRoot "ucrt64\bin"
$MsysBin = Join-Path $MsysRoot "usr\bin"

function Refresh-Path {
    $MachinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$MachinePath;$UserPath;$UcrtBin;$MsysBin"
}

function Install-WingetPackage([string]$Id, [string]$Name, [string[]]$Extra = @()) {
    Write-Host "Installing $Name..."
    & winget install --exact --id $Id --accept-package-agreements --accept-source-agreements --silent @Extra
    if ($LASTEXITCODE -ne 0) { throw "winget could not install $Name ($Id)." }
    Refresh-Path
}

function Require-Path([string]$Path, [string]$Message) {
    if (-not (Test-Path $Path)) { throw $Message }
}

function Require-Command([string]$Name, [string]$Message) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) { throw $Message }
}

Require-Command git "Git was not found. Install Git for Windows and reopen PowerShell."
Require-Command winget "Windows Package Manager (winget) was not found. Install App Installer from Microsoft Store."

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Install-WingetPackage "Kitware.CMake" "CMake"
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Install-WingetPackage "Python.Python.3.12" "Python 3.12"
} else {
    & py -3.12 --version *> $null
    if ($LASTEXITCODE -ne 0) { Install-WingetPackage "Python.Python.3.12" "Python 3.12" }
}
if (-not (Test-Path (Join-Path $MsysRoot "usr\bin\bash.exe"))) {
    Install-WingetPackage "MSYS2.MSYS2" "MSYS2" @("--location", $MsysRoot)
}

Refresh-Path
Require-Command cmake "CMake was installed but is not available. Reopen PowerShell and rerun this script."
Require-Command py "Python was installed but is not available. Reopen PowerShell and rerun this script."
Require-Path (Join-Path $MsysRoot "usr\bin\bash.exe") "MSYS2 is unavailable. Set MSYS2_ROOT if it is installed outside C:\msys64."

if (-not (Test-Path (Join-Path $UcrtBin "gcc.exe")) -or
    -not (Test-Path (Join-Path $MsysBin "make.exe")) -or
    -not (Test-Path (Join-Path $MsysBin "pkg-config.exe"))) {
    Write-Host "Installing the MSYS2 UCRT64 build toolchain..."
    $Bash = Join-Path $MsysRoot "usr\bin\bash.exe"
    & $Bash -lc "pacman -Syu --noconfirm"
    if ($LASTEXITCODE -ne 0) { throw "MSYS2 update failed. Rerun this script once if MSYS2 requested a restart." }
    & $Bash -lc "pacman -S --needed --noconfirm make pkgconf mingw-w64-ucrt-x86_64-toolchain"
    if ($LASTEXITCODE -ne 0) { throw "MSYS2 toolchain installation failed." }
}

Require-Path (Join-Path $UcrtBin "gcc.exe") "The MSYS2 UCRT64 GCC toolchain is missing. See README.md."
Require-Path (Join-Path $UcrtBin "mingw32-make.exe") "The MSYS2 UCRT64 make tool is missing. See README.md."
Require-Path (Join-Path $MsysBin "make.exe") "MSYS2 make is missing. See README.md."
Require-Path (Join-Path $MsysBin "pkg-config.exe") "MSYS2 pkg-config is missing. See README.md."

if (-not (Test-Path (Join-Path $GameRepo "python\gravity_lab"))) {
    if ($GameRepo -ne (Join-Path $Root "gravity-lab")) {
        throw "Gravity Lab Python package is missing under GRAVITY_LAB_REPO."
    }
    Write-Host "Initializing Gravity Lab submodule..."
    if (Test-Path (Join-Path $Root ".git")) {
        & git -C $Root submodule update --init --recursive
    } else {
        & git clone https://github.com/vdeviatkov/gravity-lab.git $GameRepo
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not initialize the Gravity Lab submodule." }
}

$BuildDir = Join-Path $Root "build-native"
$OutputDir = Join-Path $GameRepo "build-classic-rl"
Write-Host "Building the native classic library, viewer, and AI Arcade..."
& cmake -S $Root -B $BuildDir -G "MinGW Makefiles" `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$(Join-Path $UcrtBin 'gcc.exe')" `
    "-DCMAKE_CXX_COMPILER=$(Join-Path $UcrtBin 'g++.exe')" `
    "-DCMAKE_MAKE_PROGRAM=$(Join-Path $UcrtBin 'mingw32-make.exe')"
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }
& cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { throw "Native build failed." }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$DownloadedDllDir = Join-Path $BuildDir "x86_64-w64-mingw32\bin"
if (Test-Path $DownloadedDllDir) {
    Copy-Item (Join-Path $DownloadedDllDir "*.dll") $OutputDir -Force
}
foreach ($RuntimeDll in @("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll")) {
    $Source = Join-Path $UcrtBin $RuntimeDll
    if (Test-Path $Source) { Copy-Item $Source $OutputDir -Force }
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    & py -3.12 -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python 3.12 virtual environment." }
}
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Could not install Python build tooling." }
& $VenvPython -m pip install -e $GameRepo -e "${Root}[test]"
if ($LASTEXITCODE -ne 0) { throw "Could not install the Python projects." }

foreach ($Required in @("gravity_lab_classic.dll", "gravity_lab_classic_viewer.exe", "gravity_lab_ai_arcade.exe")) {
    Require-Path (Join-Path $OutputDir $Required) "Native output is missing: $Required"
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Smoke training: .\.venv\Scripts\gravity-lab-rl.exe train --duration-seconds 60"
Write-Host "AI Arcade:     .\.venv\Scripts\gravity-lab-rl.exe arcade"
