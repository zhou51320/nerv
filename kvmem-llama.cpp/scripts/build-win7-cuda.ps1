Param(
  [int]$Jobs = [int]::Parse($(if ($env:NUMBER_OF_PROCESSORS) { $env:NUMBER_OF_PROCESSORS } else { '1' })),
  [switch]$Clean,
  [string]$CudaArch = '75',
  [string]$Generator = 'Ninja',
  [string]$BuildDir = ''
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Out = Join-Path (Join-Path (Join-Path (Join-Path (Join-Path $Root '..') 'EVA_BACKEND') 'x86_64') 'win7') 'cuda/kvmem-llama.cpp'
$Bdir = if ($BuildDir) { $BuildDir } else { Join-Path $Root 'build-win7-cuda' }

function Require-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "$Name not found in PATH"
  }
}

Require-Command 'cmake'
Require-Command 'nvcc'
Require-Command 'cl'
if ($Generator -eq 'Ninja') { Require-Command 'ninja' }

$nvcc = (Get-Command nvcc).Source
$yy = Join-Path (Join-Path (Join-Path (Join-Path $Root '..') 'third_party') 'YY-Thunks') 'objs/x64/YY_Thunks_for_Win7.obj'
if (-not (Test-Path $yy)) {
  throw "YY-Thunks object not found: $yy"
}

if ($Clean -and (Test-Path $Bdir)) {
  Remove-Item -Recurse -Force $Bdir
}
New-Item -ItemType Directory -Force -Path $Bdir | Out-Null

# CUDA 11.x + MSVC v142, with the same Win7 tuning used by the verified
# llama.cpp build. Static cudart avoids the Win8-only api-set imports in
# cudart64_110.dll; YY-Thunks covers remaining runtime API references.
$cudaCompat = '--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -Xcompiler=/D_WIN32_WINNT=0x0601 -Xcompiler=/DWINVER=0x0601'
$cmakeArgs = @(
  '-S', $Root, '-B', $Bdir, '-G', $Generator,
  '-DCMAKE_BUILD_TYPE=Release',
  '-DCMAKE_CUDA_STANDARD=17', '-DCMAKE_CUDA_STANDARD_REQUIRED=ON',
  "-DCMAKE_CUDA_COMPILER:FILEPATH=$nvcc",
  '-DCMAKE_CUDA_HOST_COMPILER:FILEPATH=cl.exe',
  "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
  "-DCMAKE_CUDA_FLAGS_INIT:STRING=$cudaCompat",
  "-DCMAKE_CUDA_FLAGS:STRING=$cudaCompat",
  '-DGGML_CUDA=ON', '-DGGML_CUDA_FA=ON', '-DGGML_CUDA_FA_ALL_QUANTS=ON',
  '-DGGML_CUDA_GRAPHS=ON', '-DGGML_CUDA_NO_VMM=ON', '-DGGML_STATIC=ON',
  # Match the known-good llama.cpp package: ship llama/ggml backend DLLs
  # alongside the executables instead of relying on static-only linkage.
  '-DBUILD_SHARED_LIBS=ON',
  '-DGGML_NATIVE=OFF', '-DGGML_WIN_VER=0x601',
  # Static MSVC/UCRT runtime is required on unmodified Windows 7 systems.
  '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded',
  '-DCMAKE_C_FLAGS=/MT /D_WIN32_WINNT=0x0601 /DWINVER=0x0601',
  '-DCMAKE_CXX_FLAGS=/MT /EHsc /D_WIN32_WINNT=0x0601 /DWINVER=0x0601',
  "-DYY_THUNKS_OBJ:FILEPATH=$yy",
  '-DLLAMA_KVMEM=ON', "-DLLAMA_KVMEM_ROOT:PATH=$Root", '-DKVMEM_ENABLE_NVME=OFF',
  '-DKVMEM_BUILD_LLAMA=ON', '-DLLAMA_BUILD_COMMON=ON', '-DLLAMA_BUILD_TOOLS=ON',
  '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_TESTS=OFF',
  '-DLLAMA_CURL=OFF', '-DLLAMA_OPENSSL=OFF'
)

Write-Host "==> Configuring KVMem Win7 CUDA sm_$CudaArch"
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed: $LASTEXITCODE" }

$targets = @('llama-kvmem-server', 'llama-kvmem-cli', 'llama-quantize')
$buildArgs = @('--build', $Bdir, '--config', 'Release', '--target') + $targets
if ($Jobs -gt 0) { $buildArgs += @('--parallel', "$Jobs") }
Write-Host "==> Building $($targets -join ', ')"
& cmake @buildArgs
if ($LASTEXITCODE -ne 0) { throw "CMake build failed: $LASTEXITCODE" }

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$names = @('llama-kvmem-server.exe', 'llama-kvmem-cli.exe', 'llama-quantize.exe')
foreach ($name in $names) {
  $hit = Get-ChildItem -Path $Bdir -Recurse -File -Filter $name | Select-Object -First 1
  if (-not $hit) { throw "Built binary missing: $name" }
  Copy-Item $hit.FullName (Join-Path $Out $name) -Force
}

Get-ChildItem -Path $Bdir -Recurse -File -Filter '*.dll' |
  Sort-Object FullName |
  ForEach-Object { Copy-Item $_.FullName (Join-Path $Out $_.Name) -Force }

# Bundle the transitive CUDA/backend DLL closure.  The previous package only
# copied DLLs emitted directly by CMake, while cuBLAS and newer ggml backends
# can load additional CUDA DLLs at process start.  Resolve names reported by
# dumpbin against the build tree and CUDA_PATH\bin, but never copy Windows
# system DLLs.
if (Get-Command dumpbin -ErrorAction SilentlyContinue) {
  $pending = [System.Collections.Generic.Queue[string]]::new()
  Get-ChildItem -Path $Out -File | Where-Object { $_.Extension -in @('.exe', '.dll') } |
    ForEach-Object { $pending.Enqueue($_.FullName) }
  $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  while ($pending.Count -gt 0) {
    $parent = $pending.Dequeue()
    if (-not $seen.Add($parent)) { continue }
    $deps = & dumpbin /nologo /dependents $parent 2>$null |
      ForEach-Object { if ($_ -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') { $Matches[1] } }
    foreach ($dep in $deps) {
      if (Test-Path (Join-Path $Out $dep)) { $pending.Enqueue((Join-Path $Out $dep)); continue }
      $src = Get-ChildItem -Path $Bdir -Recurse -File -Filter $dep -ErrorAction SilentlyContinue |
        Select-Object -First 1
      if (-not $src -and $env:CUDA_PATH) {
        $src = Get-ChildItem -Path (Join-Path $env:CUDA_PATH 'bin') -File -Filter $dep -ErrorAction SilentlyContinue |
          Select-Object -First 1
      }
      if ($src) {
        $dest = Join-Path $Out $src.Name
        Copy-Item $src.FullName $dest -Force
        $pending.Enqueue($dest)
        Write-Host "Bundled dependency $($src.Name) for $(Split-Path $parent -Leaf)"
      }
    }
  }
}

if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH 'bin'))) {
  foreach ($pattern in @('cublas64_*.dll', 'cublasLt64_*.dll')) {
    $cudaDll = Get-ChildItem -Path (Join-Path $env:CUDA_PATH 'bin') -File -Filter $pattern |
      Sort-Object Name | Select-Object -First 1
    if ($cudaDll) {
      Copy-Item $cudaDll.FullName (Join-Path $Out $cudaDll.Name) -Force
    } else {
      Write-Warning "$pattern not found under CUDA_PATH\bin"
    }
  }
} else {
  Write-Warning 'CUDA_PATH is not set; bundle cublas64_11.dll and cublasLt64_11.dll manually before shipping.'
}

Write-Host "Done. Artifacts under $Out"
