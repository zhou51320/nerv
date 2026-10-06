#requires -Version 5.1
[CmdletBinding()]
param(
  [string]$Source = '',
  [string]$BuildDir = '',
  [string]$PackageDir = ''
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Source) { $Source = Join-Path $Root 'third_party\Strata' }
if (-not $BuildDir) { $BuildDir = Join-Path $Root 'build-strata-win7-cuda' }
$Source = (Resolve-Path $Source).Path

$required = @('CMakeLists.txt', 'serve\server.py', 'serve\winjob.py', 'src\kernels\ngram.cpp', 'src\core\expert_source.cpp')
foreach ($f in $required) { if (-not (Test-Path (Join-Path $Source $f))) { throw "Missing Strata file: $f" } }
$cmake = Get-Content (Join-Path $Source 'CMakeLists.txt') -Raw
$winjob = Get-Content (Join-Path $Source 'serve\winjob.py') -Raw
$ngram = Get-Content (Join-Path $Source 'src\kernels\ngram.cpp') -Raw
$pool = Get-Content (Join-Path $Source 'src\kernels\cpu\pool.cpp') -Raw
$poolWin = Get-Content (Join-Path $Source 'src\kernels\cpu\pool_affinity_win.hpp') -Raw
if ($cmake -notmatch 'option\(STRATA_WIN7') { throw 'STRATA_WIN7 profile is missing from CMakeLists.txt' }
if ($cmake -notmatch 'CMAKE_CUDA_RUNTIME_LIBRARY Static') { throw 'Win7 static cudart policy is missing' }
if ($cmake -notmatch '75-real') { throw 'Win7 sm75-real policy is missing' }
if ($winjob -match '(?m)^\s*_k32\.SetProcessInformation\.argtypes') { throw 'SetProcessInformation is still a hard ctypes import' }
if ($ngram -match '(?m)^\s*if \(n > 0\).*PrefetchVirtualMemory\(') { throw 'ngram still directly imports PrefetchVirtualMemory' }
if ($ngram -notmatch 'GetProcAddress\(') { throw 'ngram runtime API probe is missing' }
foreach ($api in @('GetThreadSelectedCpuSets', 'SetThreadSelectedCpuSets', 'GetSystemCpuSetInformation')) {
  foreach ($text in @($pool, $poolWin)) {
    foreach ($line in ($text -split "`r?`n")) {
      if ($line -match "\b$api\s*\(" -and $line -notmatch 'GetProcAddress') {
        throw "Win7 audit: hard CPU Set API reference remains: $api"
      }
    }
  }
}

if (Test-Path (Join-Path $BuildDir 'CMakeCache.txt')) {
  $cache = Get-Content (Join-Path $BuildDir 'CMakeCache.txt') -Raw
  foreach ($check in @(
      @{ pattern = 'STRATA_WIN7:(BOOL|STRING)=ON'; label = 'STRATA_WIN7=ON' },
      @{ pattern = 'CMAKE_CUDA_ARCHITECTURES:(BOOL|STRING)=.*75-real'; label = 'CUDA architecture 75-real' },
      @{ pattern = 'CMAKE_CUDA_RUNTIME_LIBRARY:(BOOL|STRING)=.*Static'; label = 'static CUDA runtime' }
    )) {
    if ($cache -notmatch $check.pattern) {
      Write-Warning "Build cache does not expose a canonical entry for $($check.label); source/configure policy remains authoritative."
    }
  }
}
if ($PackageDir) {
  $PackageDir = (Resolve-Path $PackageDir).Path
  foreach ($name in @('strata.exe', 'strata-device.exe', 'BUILD.json')) {
    if (-not (Test-Path (Join-Path $PackageDir $name) -PathType Leaf)) { throw "Package missing: $name" }
  }
  $dlls = @(Get-ChildItem -LiteralPath $PackageDir -Recurse -File -Filter '*.dll')
  foreach ($dll in $dlls) {
    if ($dll.Name -notmatch '^(cublas64_|cublasLt64_|cudart64_).*\.dll$') {
      throw "Strata package contains forbidden DLL: $($dll.FullName)"
    }
  }
  foreach ($pattern in @('cublas64_*.dll', 'cublasLt64_*.dll', 'cudart64_*.dll')) {
    if (-not (Get-ChildItem -LiteralPath $PackageDir -File -Filter $pattern)) {
      throw "Strata package missing required CUDA runtime DLL: $pattern"
    }
  }
  $build = Get-Content (Join-Path $PackageDir 'BUILD.json') -Raw | ConvertFrom-Json
  if ($build.profile -ne 'win7-cuda' -or $build.system -ne 'win' -or $build.device -ne 'cuda') {
    throw 'BUILD.json does not identify the win/cuda Strata profile'
  }
}
Write-Host "Strata Win7 source audit passed: $Source"
Write-Host 'Runtime features retained: C++ engine, Python HTTP/OpenAI/Anthropic/MCP/image service.'
Write-Host 'CUDA driver nvcuda.dll is intentionally not bundled; it must come from the installed Win7 NVIDIA driver.'
Write-Host 'Native compilation and Win7 GPU startup still require the Windows qualification host.'
