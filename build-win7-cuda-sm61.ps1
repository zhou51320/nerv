Param(
  [int]$Jobs        = [int]::Parse($env:NUMBER_OF_PROCESSORS),
  [switch]$Clean,
  [string]$LlamaSrc = '',
  [string]$CudaArch = '61'
)

$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSNativeCommandUseErrorActionPreference = $true
}

function Test-Cmd([string]$name) {
  return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Invoke-Native([string]$exe,[string[]]$cmdArgs) {
  & $exe @cmdArgs
  if ($LASTEXITCODE -ne 0) {
    throw "$exe failed with exit code $LASTEXITCODE. Args: $($cmdArgs -join ' ')"
  }
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

function Resolve-LlamaSrc([string]$cli,[string]$root) {
  if ($cli) { return $cli }
  $envVal = [Environment]::GetEnvironmentVariable('LLAMA_SRC')
  if ($envVal) { return $envVal }

  $candidates = @(
    (Join-Path $root 'llama.cpp'),
    (Join-Path (Join-Path $root 'external') 'llama.cpp')
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
  $candidates = @(
    (Join-Path $bdir "$name.exe"),
    (Join-Path (Join-Path $bdir 'bin') "$name.exe"),
    (Join-Path (Join-Path $bdir 'Release') "$name.exe"),
    (Join-Path (Join-Path (Join-Path $bdir 'bin') 'Release') "$name.exe"),
    (Join-Path $bdir $name),
    (Join-Path (Join-Path $bdir 'bin') $name)
  )
  foreach ($c in $candidates) {
    if (Test-Path $c) {
      Copy-Item $c -Destination $outdir -Force
      Write-Host "Copied $(Split-Path $c -Leaf) -> $outdir"
      return $true
    }
  }
  Write-Warning "Could not locate built binary '$name' under $bdir"
  return $false
}

function Find-BinaryPath([string]$bdir,[string[]]$names) {
  foreach ($name in $names) {
    $direct = @(
      (Join-Path $bdir "$name.exe"),
      (Join-Path (Join-Path $bdir 'bin') "$name.exe"),
      (Join-Path (Join-Path $bdir 'Release') "$name.exe"),
      (Join-Path (Join-Path (Join-Path $bdir 'bin') 'Release') "$name.exe")
    )
    foreach ($p in $direct) {
      if (Test-Path $p) { return $p }
    }
  }

  $exeList = Get-ChildItem -Recurse -File -Path $bdir -Filter *.exe -ErrorAction SilentlyContinue
  foreach ($name in $names) {
    $hit = $exeList | Where-Object { $_.Name -ieq "$name.exe" } | Select-Object -First 1
    if ($hit) { return $hit.FullName }
  }
  return $null
}

function Copy-BinaryByAliases([string]$bdir,[string[]]$names,[string]$destName,[string]$outdir) {
  $src = Find-BinaryPath $bdir $names
  if (-not $src) { return $false }
  New-Item -ItemType Directory -Force -Path $outdir | Out-Null
  $dest = Join-Path $outdir "$destName.exe"
  Copy-Item $src -Destination $dest -Force
  Write-Host "Copied $(Split-Path $src -Leaf) -> $dest"
  return $true
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
    'GGML_CUDA',
    'GGML_WIN_VER',
    'LLAMA_BUILD_COMMON',
    'LLAMA_BUILD_TOOLS',
    'LLAMA_BUILD_SERVER',
    'LLAMA_BUILD_EXAMPLES',
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
$src  = Resolve-LlamaSrc $LlamaSrc $ROOT

if (-not $src) {
  throw "llama.cpp source not found. Provide -LlamaSrc or set LLAMA_SRC or place repo at .\llama.cpp or .\external\llama.cpp."
}
if (-not (Test-Path (Join-Path $src 'CMakeLists.txt'))) {
  throw "Invalid llama.cpp source path: $src (CMakeLists.txt not found)"
}

if (-not (Test-Cmd 'cmake')) { throw "cmake not found in PATH." }
if (-not (Test-Cmd 'nvcc')) { throw "nvcc not found in PATH. Install CUDA toolkit and open a shell with CUDA env." }
if (-not (Test-Cmd 'gcc')) { throw "gcc not found in PATH. MinGW GCC >= 12 is recommended for Win7 builds." }
if (-not (Test-Cmd 'mingw32-make') -and -not (Test-Cmd 'make')) {
  throw "mingw32-make/make not found in PATH. Install MinGW make tools."
}

$buildRoot = Join-Path $ROOT ("build-$arch-win7")
$bdir = Join-Path (Join-Path $buildRoot 'llama.cpp') 'cuda'
if ($Clean) {
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $bdir
}

$compileFlags = '-fopenmp -mthreads -D_WIN32_WINNT=0x0601'
$linkFlags = '-static -static-libgcc -static-libstdc++ -fopenmp -Wl,-s -Wl,--gc-sections -mthreads -lpthread'

$defs = @(
  '-DBUILD_SHARED_LIBS=OFF',
  '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
  '-DCMAKE_BUILD_TYPE=Release',
  '-DCMAKE_CUDA_FLAGS:STRING=-allow-unsupported-compiler',
  "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
  "-DCMAKE_C_FLAGS:STRING=$compileFlags",
  "-DCMAKE_CXX_FLAGS:STRING=$compileFlags",
  "-DCMAKE_EXE_LINKER_FLAGS:STRING=$linkFlags",
  "-DCMAKE_SHARED_LINKER_FLAGS:STRING=$linkFlags",
  "-DCMAKE_MODULE_LINKER_FLAGS:STRING=$linkFlags",
  '-DCMAKE_OBJECT_PATH_MAX=196',
  '-DGGML_WIN_VER=0x601',
  '-DGGML_AVX512=OFF',
  '-DGGML_CUDA=ON',
  '-DGGML_NATIVE=OFF',
  '-DLLAMA_BUILD_COMMON=ON',
  '-DLLAMA_BUILD_TOOLS=ON',
  '-DLLAMA_CURL=OFF',
  '-DLLAMA_OPENSSL=OFF',
  '-DLLAMA_BUILD_TESTS=OFF',
  '-DLLAMA_BUILD_EXAMPLES=OFF',
  '-DLLAMA_BUILD_SERVER=ON'
)

New-Item -ItemType Directory -Force -Path $bdir | Out-Null

$configureArgs = @('-S', $src, '-B', $bdir, '-G', 'MinGW Makefiles') + $defs
Write-Host "==> ARCH=$arch OUT_OS=win7 DEVICE=cuda PROJECT=llama.cpp CMAKE_CUDA_ARCHITECTURES=$CudaArch BUILD_DIR=$bdir"
Invoke-Native 'cmake' $configureArgs
Show-KeyCacheValues $bdir

$buildArgs = @('--build', $bdir, '--config', 'Release')
if ($Jobs -gt 0) { $buildArgs += @('--parallel', "$Jobs") }
Invoke-Native 'cmake' $buildArgs

$outDir = Join-Path (Join-Path (Join-Path (Join-Path $OUT $arch) 'win7') 'cuda') 'llama.cpp'
$okServer = Copy-Binary $bdir 'llama-server' $outDir
if (-not $okServer) {
  $okServer = Copy-BinaryByAliases $bdir @('llama-server','server') 'llama-server' $outDir
}
$okQuant  = Copy-Binary $bdir 'llama-quantize' $outDir
if (-not $okQuant) {
  $okQuant = Copy-BinaryByAliases $bdir @('llama-quantize','quantize') 'llama-quantize' $outDir
}
if (-not $okServer -or -not $okQuant) {
  Write-Warning "Built exe files found under ${bdir}:"
  Get-ChildItem -Recurse -File -Path $bdir -Filter *.exe -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Warning "  $($_.FullName)"
  }
  throw "Build completed but required binaries are missing under $outDir"
}

Write-Host "Done. Artifacts under: $outDir"
