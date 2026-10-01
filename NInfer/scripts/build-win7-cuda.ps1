#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Win7SystemDir,
    # Explicit permission to replace only directories marked as owned by this script.
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not [Environment]::Is64BitProcess -or $env:OS -ne 'Windows_NT') {
    throw 'Run in an x64 Windows PowerShell developer shell (MSVC v142 14.29).'
}
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$BuildDir = Join-Path $Root 'build-win7-cuda'
$Out = [IO.Path]::GetFullPath((Join-Path $Root '..\EVA_BACKEND\x86_64\win\cuda\NInfer'))
$Win7SystemDir = (Resolve-Path -LiteralPath $Win7SystemDir).Path
$yy = [IO.Path]::GetFullPath((Join-Path $Root '..\third_party\YY-Thunks\objs\x64\YY_Thunks_for_Win7.obj'))
$marker = '.ninfer-win7-owned'

function Require-Tool([string]$Name) {
    $cmd = Get-Command $Name -CommandType Application -ErrorAction Stop
    return $cmd.Source
}
function Invoke-Tool([string]$File, [string[]]$Arguments) {
    # PS 5.1's legacy native argument passing can strip the quotes inside the
    # CMake /MAP flag. Quote argv explicitly while inheriting the console streams.
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $File
    $start.UseShellExecute = $false
    $start.Arguments = (($Arguments | ForEach-Object {
        '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
    }) -join ' ')
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    try {
        if (-not $process.Start()) { throw "Could not start $File" }
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { throw "$File failed with exit code $($process.ExitCode)" }
    } finally { $process.Dispose() }
}
function Capture-Tool([string]$File, [string[]]$Arguments) {
    $output = @(& $File @Arguments 2>&1 | ForEach-Object { "$_" })
    if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code ${LASTEXITCODE}: $($output -join '`n')" }
    return $output
}
function Check-NoReparse([string]$Path) {
    $cursor = $Path
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing reparse-point path: $cursor"
            }
        }
        $cursor = Split-Path -Path $cursor -Parent
    }
    if (Test-Path -LiteralPath $Path -PathType Container) {
        foreach ($item in (Get-ChildItem -LiteralPath $Path -Recurse -Force)) {
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing reparse point: $($item.FullName)" }
        }
    }
}
function Check-OwnedDirectory([string]$Path, [string]$Kind, [bool]$RequireClean) {
    Check-NoReparse $Path
    if (-not (Test-Path -LiteralPath $Path)) { return }
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { throw "Not a directory: $Path" }
    if (@(Get-ChildItem -LiteralPath $Path -Force).Count -eq 0) { return }
    $tag = Join-Path $Path $marker
    if (-not (Test-Path -LiteralPath $tag -PathType Leaf) -or
        (Get-Content -LiteralPath $tag -Raw).Trim() -cne "ninfer-win7-cuda/$Kind/v1") {
        throw "Refusing unowned nonempty directory (even with -Clean): $Path"
    }
    if ($RequireClean -and -not $Clean) { throw "Nonempty output: $Path. Use -Clean to explicitly replace this script's own package." }
}

# Fail before configuring or changing any directory.
foreach ($managed in @($BuildDir, $Out)) {
    if ($Win7SystemDir -ieq $managed -or $Win7SystemDir.StartsWith($managed.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Keep the target System32 baseline outside managed build/package directories: $Win7SystemDir"
    }
}
Check-OwnedDirectory $BuildDir 'build' $false
Check-OwnedDirectory $Out 'package' $true
$cmake = Require-Tool 'cmake.exe'
$null = Require-Tool 'ninja.exe'
$cl = Require-Tool 'cl.exe'
$link = Require-Tool 'link.exe'
$dumpbin = Require-Tool 'dumpbin.exe'
$nvcc = Require-Tool 'nvcc.exe'
$cudaRoot = Split-Path (Split-Path $nvcc -Parent) -Parent
$cudaBin = Join-Path $cudaRoot 'bin'
$cuobjdump = Join-Path $cudaBin 'cuobjdump.exe'
if (-not (Test-Path -LiteralPath $cuobjdump -PathType Leaf)) { throw "Missing CUDA cuobjdump: $cuobjdump" }
if (-not (Test-Path -LiteralPath $yy -PathType Leaf)) { throw "Missing YY-Thunks object: $yy" }
if (-not $env:VCToolsVersion -or $env:VCToolsVersion.Trim() -notmatch '^14\.29\.\d+\\?$' -or -not $env:VCToolsInstallDir) {
    throw 'Activate vcvars64.bat -vcvars_ver=14.29; no unsupported-compiler override is used.'
}
$expectedTools = [IO.Path]::GetFullPath((Join-Path $env:VCToolsInstallDir 'bin\Hostx64\x64'))
foreach ($tool in @($cl, $link, $dumpbin)) {
    if ((Split-Path $tool -Parent).TrimEnd('\') -ine $expectedTools.TrimEnd('\')) {
        throw "Tool is not from the selected x64 v142 directory: $tool"
    }
    $v = (Get-Item -LiteralPath $tool).VersionInfo
    # cl uses 19.29; linker/dumpbin use 14.29.
    $major = if ($tool -eq $cl) { 19 } else { 14 }
    if ($v.FileMajorPart -ne $major -or $v.FileMinorPart -ne 29) { throw "Not a v142 14.29 tool: $tool ($($v.FileVersion))" }
}
foreach ($name in @('CL', '_CL_', 'LINK', '_LINK_', 'CFLAGS', 'CXXFLAGS', 'CUDAFLAGS', 'LDFLAGS', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS')) {
    if ([Environment]::GetEnvironmentVariable($name)) { throw "Unset $name; ambient flags can override the controlled Win7/CRT configuration." }
}
$nvccVersion = (Capture-Tool $nvcc @('--version')) -join "`n"
if ($nvccVersion -notmatch '\brelease 11\.7,\s+V11\.7\.\d+\b') { throw "CUDA 11.7.x required, got: $nvccVersion" }
$cmakeVersion = (Capture-Tool $cmake @('--version')) -join "`n"
if ($cmakeVersion -notmatch 'cmake version (\d+\.\d+\.\d+)' -or [version]$Matches[1] -lt [version]'3.28.0') {
    throw "CMake >= 3.28 required: $cmakeVersion"
}
$baseline = Join-Path $Win7SystemDir 'kernel32.dll'
if (-not (Test-Path -LiteralPath $baseline -PathType Leaf)) { throw 'Missing Win7 SP1 x64 System32 baseline' }
$v = (Get-Item -LiteralPath $baseline).VersionInfo
if ($v.FileMajorPart -ne 6 -or $v.FileMinorPart -ne 1 -or $v.FileBuildPart -ne 7601) { throw 'Win7SystemDir must be a target Windows 7 SP1 baseline' }

if ($Clean -and (Test-Path -LiteralPath $BuildDir)) { Remove-Item -LiteralPath $BuildDir -Recurse -Force }
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
Set-Content -LiteralPath (Join-Path $BuildDir $marker) -Value 'ninfer-win7-cuda/build/v1' -Encoding ASCII
$bin = Join-Path $BuildDir 'bin'
$map = Join-Path $BuildDir 'ninfer.map'
$win7Flags = '/D_WIN32_WINNT=0x0601 /DWINVER=0x0601 /DNTDDI_VERSION=0x06010000'
$cudaFlags = '-D_WIN32_WINNT=0x0601 -DWINVER=0x0601 -DNTDDI_VERSION=0x06010000 -Xcompiler=/MD'
$cmakeArgs = @(
    '-S', $Root, '-B', $BuildDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE:STRING=Release', '-DBUILD_SHARED_LIBS:BOOL=OFF',
    '-DNINFER_WIN7:BOOL=ON', '-DNINFER_TEXT_ONLY:BOOL=ON',
    '-DNINFER_BUILD_APPS:BOOL=ON', '-DNINFER_BUILD_SERVER:BOOL=OFF',
    '-DNINFER_DISABLE_NVTX:BOOL=ON', '-DBUILD_TESTING:BOOL=OFF', '-DNINFER_BUILD_BENCHMARKS:BOOL=OFF',
    '-DCMAKE_CXX_STANDARD:STRING=20', '-DCMAKE_CXX_STANDARD_REQUIRED:BOOL=ON',
    '-DCMAKE_CUDA_STANDARD:STRING=17', '-DCMAKE_CUDA_STANDARD_REQUIRED:BOOL=ON',
    '-DCMAKE_CUDA_ARCHITECTURES:STRING=75-real',
    '-DCMAKE_CUDA_RUNTIME_LIBRARY:STRING=Static', '-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreadedDLL',
    "-DCMAKE_C_COMPILER:FILEPATH=$cl", "-DCMAKE_CXX_COMPILER:FILEPATH=$cl",
    "-DCMAKE_LINKER:FILEPATH=$link", "-DCMAKE_CUDA_HOST_COMPILER:FILEPATH=$cl",
    "-DCMAKE_CUDA_COMPILER:FILEPATH=$nvcc", "-DCUDAToolkit_ROOT:PATH=$cudaRoot",
    "-DCMAKE_C_FLAGS:STRING=/MD $win7Flags", "-DCMAKE_CXX_FLAGS:STRING=/MD /EHsc $win7Flags",
    "-DCMAKE_CUDA_FLAGS:STRING=$cudaFlags",
    "-DCMAKE_EXE_LINKER_FLAGS:STRING=/SUBSYSTEM:CONSOLE,6.01 /OSVERSION:6.1 /INCREMENTAL:NO /MAP:`"$($map.Replace('\', '/'))`"",
    "-DNINFER_WIN7_THUNKS:FILEPATH=$yy", "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=$bin"
)
Write-Host 'Configuring NInfer: Win7 / CUDA 11.7 / v142 14.29 / sm_75 SASS / text CLI'
Invoke-Tool $cmake $cmakeArgs
Invoke-Tool $cmake @('--build', $BuildDir, '--config', 'Release', '--target', 'ninfer', '-j')
$exe = Join-Path $bin 'ninfer.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Built CLI missing: $exe" }
if (-not (Test-Path -LiteralPath $map -PathType Leaf) -or
    (Get-Content -LiteralPath $map -Raw) -notmatch 'YY_Thunks_for_Win7') {
    throw 'The final ninfer.exe link map does not contain YY_Thunks_for_Win7; wire NINFER_WIN7_THUNKS into the CLI link.'
}
$elf = (Capture-Tool $cuobjdump @('--list-elf', $exe)) -join "`n"
if ($elf -notmatch '\bsm_75\b' -or $elf -match '\bsm_(?!75\b)[0-9]+[a-z]?\b') {
    throw "Expected only sm_75 device code in ninfer.exe: $elf"
}
$ptx = (Capture-Tool $cuobjdump @('--list-ptx', $exe)) -join "`n"
if ($ptx -match '(?im)^\s*PTX file\s+\d+\s*:') { throw "Unexpected PTX in the Win7 executable: $ptx" }

# Stage in the owned build directory. Do not remove a previous package until the
# new package has passed validation. Never enumerate or alter sibling backends.
$stage = Join-Path $BuildDir ('package-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stage | Out-Null
Set-Content -LiteralPath (Join-Path $stage $marker) -Value 'ninfer-win7-cuda/package/v1' -Encoding ASCII
Copy-Item -LiteralPath $exe -Destination $stage
$external = @(
    'kernel32.dll', 'kernelbase.dll', 'ntdll.dll', 'advapi32.dll', 'bcrypt.dll', 'crypt32.dll',
    'cryptbase.dll', 'user32.dll', 'gdi32.dll', 'shell32.dll', 'shlwapi.dll', 'ole32.dll',
    'oleaut32.dll', 'ws2_32.dll', 'mswsock.dll', 'iphlpapi.dll', 'dnsapi.dll', 'secur32.dll',
    'sspicli.dll', 'version.dll', 'winmm.dll', 'psapi.dll', 'dbghelp.dll', 'powrprof.dll',
    'setupapi.dll', 'cfgmgr32.dll', 'rpcrt4.dll', 'normaliz.dll', 'msvcrt.dll', 'imm32.dll', 'nvcuda.dll'
)
$crt = '^(api-ms-win-crt-[a-z0-9-]+|ucrtbase|msvcp140(?:_[a-z0-9]+)?|vcruntime140(?:_1)?|vcomp140|concrt140)\.dll$'
$pending = [Collections.Generic.Queue[string]]::new()
$pending.Enqueue((Join-Path $stage 'ninfer.exe'))
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
while ($pending.Count -gt 0) {
    $parent = $pending.Dequeue()
    if (-not $seen.Add($parent)) { continue }
    # /imports includes delay imports. /dependents is also checked by the verifier.
    $deps = @((Capture-Tool $dumpbin @('/nologo', '/imports', $parent)) | ForEach-Object {
        if ($_ -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') { $Matches[1] }
    } | Select-Object -Unique)
    if ($deps.Count -eq 0) { throw "No parseable DLL imports: $parent" }
    foreach ($dep in $deps) {
        if ($dep -match '^(cudart64_.*|api-ms-win-core-.*|ext-ms-.*|nvToolsExt.*)\.dll$') { throw "Forbidden import: $parent -> $dep" }
        if ($dep -in $external -or $dep -match $crt) { continue }
        $dest = Join-Path $stage $dep
        if (-not (Test-Path -LiteralPath $dest -PathType Leaf)) {
            $candidates = @(@((Join-Path $bin $dep), (Join-Path $cudaBin $dep)) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
            if ($candidates.Count -ne 1) { throw "Expected one controlled dependency source for $dep, found $($candidates.Count)" }
            Copy-Item -LiteralPath $candidates[0] -Destination $dest
        }
        $pending.Enqueue($dest)
    }
}
foreach ($name in @('WIN7-RUNTIME.txt', 'verify-win7-package.ps1', 'smoke-win7-package.ps1')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $stage
}
& (Join-Path $PSScriptRoot 'verify-win7-package.ps1') -PackageDir $stage -Win7SystemDir $Win7SystemDir
@(
    'NInfer Win7 text-only CLI; model qwen3.6-27b/groupwise-int; .ninfer v2',
    'CUDA architecture: 75-real; static cudart; /MD; no graphs; no MTP; no server; no NVTX',
    "Compiler: $cl", "VCToolsVersion: $($env:VCToolsVersion)", $nvccVersion,
    "YY-Thunks final-link evidence: $map", "Win7 import baseline: $Win7SystemDir",
    'Static package checks passed. Native Win7/GPU/model smoke has NOT been run by the build script.'
) | Set-Content -LiteralPath (Join-Path $stage 'BUILD-INFO.txt') -Encoding UTF8
Get-ChildItem -LiteralPath $stage -File | Select-Object -ExpandProperty Name | Sort-Object |
    Set-Content -LiteralPath (Join-Path $stage 'PACKAGE-MANIFEST.txt') -Encoding UTF8

# Recheck immediately before the only destructive package operation.
Check-OwnedDirectory $Out 'package' $true
if (Test-Path -LiteralPath $Out) { Remove-Item -LiteralPath $Out -Recurse -Force }
New-Item -ItemType Directory -Path (Split-Path $Out -Parent) -Force | Out-Null
Move-Item -LiteralPath $stage -Destination $Out
Write-Host "Package created: $Out"
Write-Host 'No CLI execution, model download or deployment test was performed. Run smoke-win7-package.ps1 on the target.'
