Param(
  [int]$Jobs        = [int]::Parse($env:NUMBER_OF_PROCESSORS),
  [switch]$Clean,
  [string]$FastllmSrc = '',
  [string]$CudaArch = '75',
  [string]$Generator = 'auto',
  [string]$Platform = 'x64'
)

$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSNativeCommandUseErrorActionPreference = $false
}

function Test-Cmd([string]$name) {
  return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Show-CMakeDiagnosticLogs([string[]]$cmdArgs) {
  $bdir = $null
  for ($i = 0; $i -lt $cmdArgs.Count; $i++) {
    if ($cmdArgs[$i] -eq '-B' -and ($i + 1) -lt $cmdArgs.Count -and $cmdArgs[0] -ne '--build') {
      $bdir = $cmdArgs[$i + 1]
      break
    }
  }
  if (-not $bdir -and $cmdArgs.Count -ge 2 -and $cmdArgs[0] -eq '--build') {
    $bdir = $cmdArgs[1]
  }
  if (-not $bdir) { return }

  $errLog = Join-Path $bdir 'CMakeFiles\CMakeError.log'
  $outLog = Join-Path $bdir 'CMakeFiles\CMakeOutput.log'
  if (Test-Path $errLog) {
    Write-Warning "---- CMakeError.log (tail) ----"
    Get-Content -LiteralPath $errLog -Tail 200 | ForEach-Object { Write-Warning $_ }
  }
  if (Test-Path $outLog) {
    Write-Host "---- CMakeOutput.log (tail) ----"
    Get-Content -LiteralPath $outLog -Tail 120 | ForEach-Object { Write-Host $_ }
  }
}

function Invoke-Native([string]$exe,[string[]]$cmdArgs) {
  Write-Host "==> Running: $exe $($cmdArgs -join ' ')"
  $output = & $exe @cmdArgs 2>&1
  $exitCode = $LASTEXITCODE
  if ($output) {
    $output | ForEach-Object { Write-Host $_ }
  }
  if ($exitCode -ne 0) {
    if ($exe -eq 'cmake') {
      Show-CMakeDiagnosticLogs $cmdArgs
      if ($output) {
        Write-Warning "---- CMake command output (tail) ----"
        $output | Select-Object -Last 120 | ForEach-Object { Write-Warning $_ }
      }
    }
    throw "$exe failed with exit code $exitCode. Args: $($cmdArgs -join ' ')"
  }
}

function Resolve-Generator([string]$requested) {
  if ($requested -and $requested -ne 'auto') {
    return $requested
  }
  if ((Test-Cmd 'ninja') -and (Test-Cmd 'cl')) {
    return 'Ninja'
  }
  if (Test-Cmd 'cl') {
    return 'Visual Studio 17 2022'
  }
  if ((Test-Cmd 'gcc') -and ((Test-Cmd 'mingw32-make') -or (Test-Cmd 'make'))) {
    return 'MinGW Makefiles'
  }
  throw "No suitable toolchain found. Need either MSVC (cl) or MinGW (gcc + make)."
}

function Resolve-Arch {
  $arch = $env:PROCESSOR_ARCHITECTURE
  if ($env:PROCESSOR_ARCHITEW6432) {
    $arch = $env:PROCESSOR_ARCHITEW6432
  }
  switch -Regex ($arch) {
    '^(AMD64|X64)$'   { return 'x86_64' }
    '^(x86|X86)$'     { return 'x86_32' }
    '^(ARM64)$'       { return 'arm64' }
    '^(ARM)$'         { return 'arm32' }
    default           { return 'x86_64' }
  }
}

function Resolve-FastllmSrc([string]$cli,[string]$root) {
  if ($cli) { return $cli }
  $envVal = [Environment]::GetEnvironmentVariable('FASTLLM_SRC')
  if ($envVal) { return $envVal }

  $candidates = @(
    (Join-Path $root 'fastllm'),
    (Join-Path (Join-Path $root 'external') 'fastllm')
  )
  foreach ($c in $candidates) {
    if (Test-Path (Join-Path $c 'CMakeLists.txt')) {
      return $c
    }
  }
  return $null
}

function Copy-Binary([string]$bdir,[string]$name,[string]$outdir) {
  New-Item -ItemType Directory -Force -Path $outdir | Out-Null
  $hit = Get-ChildItem -Recurse -File -Path $bdir -Filter "$name.exe" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\.release\\.' } | Select-Object -First 1
  if (-not $hit) { return $false }
  Copy-Item $hit.FullName -Destination $outdir -Force
  Write-Host "Copied $($hit.FullName) -> $outdir"
  return $true
}

function Get-BinaryPath([string]$bdir,[string]$name) {
  $hit = Get-ChildItem -Recurse -File -Path $bdir -Filter "$name.exe" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\.release\\.' } | Select-Object -First 1
  if ($hit) { return $hit.FullName }
  return $null
}

function Show-KeyCacheValues([string]$bdir) {
  $cache = Join-Path $bdir 'CMakeCache.txt'
  if (-not (Test-Path $cache)) {
    Write-Warning "CMakeCache.txt not found under $bdir"
    return
  }
  $keys = @(
    'CMAKE_GENERATOR',
    'CMAKE_BUILD_TYPE',
    'CMAKE_CUDA_ARCHITECTURES',
    'USE_CUDA',
    'USE_NUMAS',
    'USE_MMAP',
    'PY_API',
    'BUILD_CLI',
    'CMAKE_CUDA_STANDARD',
    'CMAKE_RUNTIME_OUTPUT_DIRECTORY'
  )
  $content = Get-Content -LiteralPath $cache
  Write-Host '==> CMake cache summary'
  foreach ($k in $keys) {
    $line = $content | Where-Object { $_ -match "^$k(:|=)" } | Select-Object -First 1
    if ($line) { Write-Host "  $line" }
  }
}

$ROOT = (Get-Location).Path
$OUT  = Join-Path $ROOT 'EVA_BACKEND'
$arch = Resolve-Arch
$src  = Resolve-FastllmSrc $FastllmSrc $ROOT

if (-not $src) {
  throw "fastllm source not found. Provide -FastllmSrc or set FASTLLM_SRC or place repo at .\fastllm or .\external\fastllm."
}
if (-not (Test-Path (Join-Path $src 'CMakeLists.txt'))) {
  throw "Invalid fastllm source path: $src (CMakeLists.txt not found)"
}

if (-not (Test-Cmd 'cmake')) { throw "cmake not found in PATH." }
$nvccCmd = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvccCmd) {
  $nvccPath = $nvccCmd.Source
} else {
  $nvccPath = $null
  if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
    $nvccFound = Get-ChildItem -Path $env:CUDA_PATH -Recurse -File -Filter nvcc.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($nvccFound) {
      $nvccPath = $nvccFound.FullName
      $nvccDir = Split-Path $nvccPath -Parent
      $env:PATH = "$nvccDir;$env:PATH"
      Write-Host "nvcc not found in PATH; recovered from CUDA_PATH: $nvccPath"
    }
  }
  if (-not $nvccPath) {
    throw "nvcc not found in PATH and not found under CUDA_PATH. Install CUDA toolkit and open a shell with CUDA env."
  }
}

$resolvedGenerator = Resolve-Generator $Generator
$isMingw = $resolvedGenerator -like 'MinGW*'
$isVsGen = $resolvedGenerator -like 'Visual Studio*'
$isNinja = $resolvedGenerator -eq 'Ninja'

if ($isMingw) {
  throw "CUDA + MinGW on Windows is not supported reliably by nvcc. Use MSVC generator."
}
if ($isVsGen -and -not (Test-Cmd 'cl')) {
  throw "MSVC generator selected but 'cl' was not found. Run in a VS Developer Command Prompt or add ilammy/msvc-dev-cmd in GitHub Actions."
}
if ($isNinja) {
  if (-not (Test-Cmd 'ninja')) { throw "Ninja generator selected but 'ninja' was not found in PATH." }
  if (-not (Test-Cmd 'cl')) { throw "Ninja + CUDA on Windows requires MSVC host compiler. 'cl' not found." }
}

$buildRoot = Join-Path $ROOT ("build-$arch-win7")
$bdir = Join-Path (Join-Path $buildRoot 'fastllm') 'cuda'
if ($Clean) {
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $bdir
}

$defs = @(
  '-DCMAKE_BUILD_TYPE=Release',
  "-DCMAKE_CUDA_COMPILER:FILEPATH=$nvccPath",
  "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
  '-DUSE_CUDA=ON',
  '-DUSE_NUMAS=OFF',
  '-DUSE_MMAP=OFF',
  '-DUSE_ROCM=OFF',
  '-DUSE_TFACC=OFF',
  '-DPY_API=OFF',
  '-DBUILD_CLI=OFF'
)

if (Test-Path $bdir) {
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $bdir
}
New-Item -ItemType Directory -Force -Path $bdir | Out-Null

$configureArgs = @('-S', $src, '-B', $bdir, '-G', $resolvedGenerator)
if ($isVsGen -and $Platform) {
  $configureArgs += @('-A', $Platform)
}
$configureArgs += $defs

Write-Host "==> ARCH=$arch OUT_OS=win7 DEVICE=cuda PROJECT=fastllm GENERATOR=$resolvedGenerator CMAKE_CUDA_ARCHITECTURES=$CudaArch BUILD_DIR=$bdir"
Invoke-Native 'cmake' $configureArgs
Show-KeyCacheValues $bdir

$buildArgs = @('--build', $bdir, '--config', 'Release')
if ($Jobs -gt 0) { $buildArgs += @('--parallel', "$Jobs") }
Invoke-Native 'cmake' $buildArgs

$outDir = Join-Path (Join-Path (Join-Path (Join-Path $OUT $arch) 'win7') 'cuda') 'fastllm'
$okMain = Copy-Binary $bdir 'main' $outDir
$okQuant = Copy-Binary $bdir 'quant' $outDir
$okApi = Copy-Binary $bdir 'fastllm-apiserver' $outDir

if (-not $okMain -or -not $okQuant -or -not $okApi) {
  Write-Warning "Built exe files found under ${bdir}:"
  Get-ChildItem -Recurse -File -Path $bdir -Filter *.exe -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Warning "  $($_.FullName)"
  }
  throw "Build completed but required binaries are missing under $outDir"
}

Write-Host "Done. Artifacts under: $outDir"
Write-Host "  main.exe  -> $((Get-BinaryPath $bdir 'main') -replace [regex]::Escape((Get-Location).Path), '.')"
Write-Host "  quant.exe -> $((Get-BinaryPath $bdir 'quant') -replace [regex]::Escape((Get-Location).Path), '.')"
Write-Host "  fastllm-apiserver.exe -> $((Get-BinaryPath $bdir 'fastllm-apiserver') -replace [regex]::Escape((Get-Location).Path), '.')"