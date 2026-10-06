#requires -Version 5.1
[CmdletBinding()]
param(
  [string]$Source = '',
  [string]$BuildDir = '',
  [string]$Win7SystemDir = '',
  [string]$OutputDir = '',
  [string]$LlamaDir = '',
  [string]$CudaArch = '75-real',
  [int]$Jobs = 0,
  [switch]$Clean,
  [switch]$CleanPackage,
  [switch]$NoPackage,
  [switch]$BuildVision,
  [switch]$SkipAudit
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Source) { $Source = Join-Path $Root 'third_party\Strata' }
if (-not $BuildDir) { $BuildDir = Join-Path $Root 'build-strata-win7-cuda' }
if (-not $OutputDir) { $OutputDir = Join-Path $Root 'artifacts\Strata' }
$Source = (Resolve-Path $Source).Path
if ($CudaArch -ne '75-real') { throw 'Strata Win7 profile requires -CudaArch 75-real' }
if (-not (Test-Path (Join-Path $Source 'CMakeLists.txt'))) { throw "Strata source missing: $Source" }
$yy = Join-Path $Root 'third_party\YY-Thunks\objs\x64\YY_Thunks_for_Win7.obj'
if (-not (Test-Path $yy)) { throw "YY-Thunks object not found: $yy" }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw 'cmake not found' }
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { throw 'ninja not found' }
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) { throw 'nvcc not found' }
if ($Jobs -le 0) { $Jobs = [Math]::Max(1, [Environment]::ProcessorCount) }

if (-not $Win7SystemDir) { Write-Warning 'Win7SystemDir not supplied; build will still compile, but PE import audit cannot qualify target exports.' }
if ($Clean -and (Test-Path $BuildDir)) { Remove-Item -LiteralPath $BuildDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$cmakeArgs = @('-S', $Source, '-B', $BuildDir, '-G', 'Ninja',
  '-DCMAKE_BUILD_TYPE=Release', '-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_WIN7=ON',
  '-DSTRATA_BUILD_TESTS=OFF', '-DSTRATA_BUILD_CONVERSATION_TESTS=OFF',
  '-DSTRATA_BUILD_BENCHMARKS=OFF', '-DSTRATA_ENABLE_HIP=OFF',
  '-DCMAKE_CUDA_ARCHITECTURES=75-real', '-DCMAKE_CUDA_STANDARD=17',
  '-DCMAKE_CUDA_STANDARD_REQUIRED=ON', '-DCMAKE_CUDA_RUNTIME_LIBRARY=Static',
  "-DSTRATA_WIN7_THUNKS=$yy", '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL')
Write-Host "==> Configuring Strata Win7 CUDA 11.x sm75-real: $Source"
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed: $LASTEXITCODE" }
& cmake --build $BuildDir --target strata strata-device --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "cmake build failed: $LASTEXITCODE" }
if ($BuildVision) {
  if (-not $LlamaDir) { throw '-LlamaDir is required with -BuildVision' }
  $LlamaDir = (Resolve-Path $LlamaDir).Path
  $VisionBuildDir = Join-Path $Root 'build-strata-win7-vision'
  if ($Clean -and (Test-Path $VisionBuildDir)) { Remove-Item -LiteralPath $VisionBuildDir -Recurse -Force }
  $visionArgs = @('-S', (Join-Path $Source 'tools\vision'), '-B', $VisionBuildDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DSTRATA_VISION_CUDA=ON', '-DSTRATA_PORTABLE=ON', '-DSTRATA_WIN7=ON',
    '-DCMAKE_CUDA_ARCHITECTURES=75-real', '-DCMAKE_CUDA_STANDARD=17', '-DCMAKE_CUDA_STANDARD_REQUIRED=ON',
    '-DCMAKE_CUDA_RUNTIME_LIBRARY=Static', "-DLLAMA_DIR=$LlamaDir",
    "-DCMAKE_EXE_LINKER_FLAGS=$yy /SUBSYSTEM:CONSOLE,6.01 /OSVERSION:6.1")
  Write-Host "==> Configuring Strata vision Win7 CUDA 11.x sm75-real"
  & cmake @visionArgs
  if ($LASTEXITCODE -ne 0) { throw "vision cmake configure failed: $LASTEXITCODE" }
  & cmake --build $VisionBuildDir --target strata-vision --parallel $Jobs
  if ($LASTEXITCODE -ne 0) { throw "vision cmake build failed: $LASTEXITCODE" }
}
if (-not $SkipAudit) { & (Join-Path $PSScriptRoot 'audit-strata-win7.ps1') -Source $Source -BuildDir $BuildDir }
if ($NoPackage) {
  Write-Host "Build complete (package skipped). Engine: $(Join-Path $BuildDir 'strata.exe')"
  exit 0
}

function Find-Binary([string]$Name) {
  $searchRoots = @($BuildDir)
  if ($BuildVision -and $Name -eq 'strata-vision') {
    $searchRoots += (Join-Path $Root 'build-strata-win7-vision')
  }
  $hit = Get-ChildItem -Path $searchRoots -Recurse -File -Filter "$Name.exe" |
    Select-Object -First 1
  if (-not $hit) { throw "Built binary not found: $Name.exe" }
  return $hit.FullName
}

$stage = Join-Path $BuildDir ('package-' + [guid]::NewGuid().ToString('N'))
$markerName = '.nerv-strata-win-package'
$markerText = 'strata-win7-cuda/v1'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Set-Content -LiteralPath (Join-Path $stage $markerName) -Value $markerText -Encoding ASCII
try {
  $binaryNames = @('strata', 'strata-device')
  if ($BuildVision) { $binaryNames += 'strata-vision' }
  foreach ($name in $binaryNames) {
    $exe = Find-Binary $name
    Copy-Item -LiteralPath $exe -Destination (Join-Path $stage "$name.exe") -Force
    $pdb = [IO.Path]::ChangeExtension($exe, '.pdb')
    if (Test-Path -LiteralPath $pdb -PathType Leaf) {
      Copy-Item -LiteralPath $pdb -Destination (Join-Path $stage "$name.pdb") -Force
    }
  }

  # These are Strata-owned, small runtime resources. Model GGUF/pack files,
  # CUDA driver DLLs and system DLLs stay outside the package. Static cudart
  # is linked into the executable; cuBLAS remains a runtime DLL like llama.cpp.
  $dataStage = Join-Path $stage 'data'
  New-Item -ItemType Directory -Force -Path $dataStage | Out-Null
  Get-ChildItem -LiteralPath (Join-Path $Source 'data') -File -Filter '*.bin' |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $dataStage $_.Name) -Force }
  if (Test-Path (Join-Path $Source 'docs\WIN7.md')) {
    Copy-Item -LiteralPath (Join-Path $Source 'docs\WIN7.md') -Destination (Join-Path $stage 'WIN7.md') -Force
  }

  if (-not $env:CUDA_PATH -or -not (Test-Path $env:CUDA_PATH)) {
    throw 'CUDA_PATH is required to bundle the cuBLAS runtime DLLs'
  }
  foreach ($pattern in @('cublas64_*.dll', 'cublasLt64_*.dll')) {
    $dll = Get-ChildItem -Path $env:CUDA_PATH -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue |
      Sort-Object FullName | Select-Object -First 1
    if (-not $dll) { throw "Required CUDA runtime DLL missing: $pattern" }
    Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $stage $dll.Name) -Force
  }

  $metaPath = Join-Path $Root 'third_party\strata-upstream.json'
  $meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
  $buildJson = [ordered]@{
    project = 'Strata'
    upstream_repository = $meta.repository
    upstream_commit = $meta.commit
    upstream_archive_sha256 = $meta.archive_sha256
    profile = 'win7-cuda'
    architecture = 'x86_64'
    system = 'win'
    compatibility_label = 'win7'
    device = 'cuda'
    cuda_architecture = '75-real'
    cuda_runtime = 'static'
    compiler = 'MSVC v142 14.29'
    yy_thunks = 'YY_Thunks_for_Win7.obj'
    optional_win8_plus_apis = 'runtime-probed-with-fallback'
    dynamic_cuda_or_system_dlls_bundled = $true
    binaries = @('strata.exe', 'strata-device.exe') + $(if ($BuildVision) { @('strata-vision.exe') } else { @() })
    version = '0.1.40'
    backend = 'cuda'
    source = 'prebuilt'
    archs = @(75)
    ptx = $true
    vision = $(if ($BuildVision) { 'gpu' } else { 'none' })
    bundled_cuda_dlls = $true
    lib_dirs = @('engine')
    resources = @('data/*.bin', 'WIN7.md')
    build_directory = [IO.Path]::GetFullPath($BuildDir)
    qualification = 'compiled-on-modern-Windows; native-Win7/GPU/model-smoke-pending'
  }
  $buildJson | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stage 'BUILD.json') -Encoding UTF8
  Get-ChildItem -LiteralPath $stage -Recurse -File -Filter '*.dll' |
    ForEach-Object {
      if ($_.Name -notmatch '^(cublas64_|cublasLt64_).*\.dll$') {
        throw "Refusing non-cuBLAS DLL in Strata package: $($_.FullName)"
      }
    }

  if (Test-Path -LiteralPath $OutputDir) {
    $owned = Test-Path -LiteralPath (Join-Path $OutputDir $markerName)
    if (-not $owned) { throw "Refusing to replace unowned Strata output directory: $OutputDir" }
    if (-not $CleanPackage) { throw "Strata output already exists; use -CleanPackage to replace the owned package: $OutputDir" }
    Remove-Item -LiteralPath $OutputDir -Recurse -Force
  }
  New-Item -ItemType Directory -Force -Path (Split-Path $OutputDir -Parent) | Out-Null
  Move-Item -LiteralPath $stage -Destination $OutputDir
  Write-Host "Package created: $OutputDir"
  & (Join-Path $PSScriptRoot 'audit-strata-win7.ps1') -Source $Source -BuildDir $BuildDir -PackageDir $OutputDir
} finally {
  if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
}
Write-Host "Build complete. Debug build remains: $BuildDir"
Write-Host 'The Python service remains separate and retains HTTP/OpenAI/Anthropic/MCP/image paths.'
