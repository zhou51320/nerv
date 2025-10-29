Param(
  [string]$Projects = 'all',      # all|llama|whisper|sd|tts (comma-separated)
  [string]$Devices  = 'auto',     # auto|cpu|vulkan|cuda|opencl|all (comma-separated)
  [int]$Jobs        = [int]::Parse($env:NUMBER_OF_PROCESSORS),
  [string]$Compiler = 'auto',     # auto|msvc|mingw
  [switch]$Clean,
  [string]$LlamaSrc = '',
  [string]$WhisperSrc = '',
  [string]$SDSrc = '',
  [string]$TTSSrc = ''
)

$ROOT      = (Get-Location).Path
$EXTERN    = Join-Path $ROOT 'external'
$BUILD     = Join-Path $ROOT 'build'
$OUT       = Join-Path $ROOT 'EVA_BACKEND'
$OS_ID     = 'win'
$script:OutOsId = $OS_ID
$script:AllDevices = @()
$script:CompilerMode = 'auto'
$script:MingwRuntimePaths = @()
$script:MingwRuntimeDlls = @(
  'libatomic-1.dll',
  'libgcc_s_seh-1.dll',
  'libgfortran-5.dll',
  'libgomp-1.dll',
  'libquadmath-0.dll',
  'libstdc++-6.dll',
  'libwinpthread-1.dll'
)
$script:RuntimeDirsCopied = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)

$Compiler = if ($Compiler) { $Compiler.Trim() } else { 'auto' }
$Compiler = $Compiler.ToLowerInvariant()
switch ($Compiler) {
  'auto' {}
  'msvc' {}
  'mingw' {}
  default { throw "Unknown compiler '$Compiler'. Use auto|msvc|mingw." }
}
$CompilerRequest = $Compiler

# Pinned references (only checked/warned)
$LLAMA_EXPECT_REF = 'b6746'
$WHISPER_EXPECT_TAG = 'v1.8.1'
$SD_EXPECT_REF = '0585e2609d26fc73cde0dd963127ae585ca62d49'
$TTS_EXPECT_REF = 'e4634fb'

function Resolve-Arch {
  $arch = $env:PROCESSOR_ARCHITECTURE
  if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 }
  switch -Regex ($arch) {
    '^(AMD64|X64)$'   { return 'x86_64' }
    '^(x86|X86)$'     { return 'x86_32' }
    '^(ARM64)$'       { return 'arm64' }
    '^(ARM)$'         { return 'arm32' }
    default           { return 'x86_64' }
  }
}

function Test-Cmd([string]$name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Can-Vulkan {
  if ($env:VULKAN_SDK) { return $true }
  if (Test-Cmd 'glslc') { return $true }
  $vkInc = Join-Path ${env:ProgramFiles} 'VulkanSDK'
  try {
    if (Test-Path $vkInc) {
      $hit = Get-ChildItem -Path $vkInc -Recurse -ErrorAction SilentlyContinue -Filter 'vulkan.h' | Select-Object -First 1
      if ($hit) { return $true }
    }
  } catch {}
  return $false
}

function Can-CUDA {
  if (Test-Cmd 'nvcc') { return $true }
  if (Test-Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA') { return $true }
  return $false
}

function Can-OpenCL {
  if (Test-Path "$env:WINDIR\System32\OpenCL.dll") { return $true }
  if (Test-Cmd 'clinfo') { return $true }
  return $false
}

function Resolve-Devices([string]$req) {
  $set = New-Object System.Collections.Generic.HashSet[string]
  if ($req -eq 'auto') {
    $set.Add('cpu') | Out-Null
    if (Can-Vulkan) { $set.Add('vulkan') | Out-Null }
  } elseif ($req -eq 'all') {
    $set.Add('cpu') | Out-Null
    if (Can-Vulkan) { $set.Add('vulkan') | Out-Null }
    if (Can-CUDA) { $set.Add('cuda') | Out-Null }
    if (Can-OpenCL) { $set.Add('opencl') | Out-Null }
  } else {
    foreach ($d in $req.Split(',')) { $null = $set.Add($d.Trim()) }
  }
  $list = New-Object System.Collections.Generic.List[string]
  foreach ($item in $set) { [void]$list.Add($item) }
  return $list.ToArray()
}

function Filter-DevicesForCompiler([string[]]$devices,[string]$compilerMode) {
  if (-not $devices) { return @() }
  if ($compilerMode -ne 'mingw') { return $devices }
  $filtered = @()
  foreach ($d in $devices) {
    if ($d -eq 'cuda') {
      Write-Host "[info] Skipping CUDA builds under MinGW compiler"
    } else {
      $filtered += $d
    }
  }
  if ($filtered.Count -eq 0) {
    Write-Warning "No devices remain after filtering CUDA for MinGW; nothing to build."
  }
  return $filtered
}

function Resolve-CompilerMode([string]$requested) {
  if ($requested -ne 'auto') { return $requested }
  $envGen = $env:CMAKE_GENERATOR
  if ($envGen -and $envGen.Trim().ToLowerInvariant() -like 'mingw*') { return 'mingw' }
  foreach ($var in @('CC','CXX')) {
    $val = [Environment]::GetEnvironmentVariable($var)
    if ($val -and $val.ToLowerInvariant().Contains('mingw')) { return 'mingw' }
  }
  if (Test-Cmd 'mingw32-make') { return 'mingw' }
  if ((Test-Cmd 'gcc') -and -not (Test-Cmd 'cl')) { return 'mingw' }
  return 'auto'
}

function Resolve-Src([string]$name,[string]$cli,[string]$envName,[string[]]$candidates) {
  if ($cli) { return $cli }
  $envVal = [Environment]::GetEnvironmentVariable($envName)
  if ($envVal) { return $envVal }
  foreach ($c in $candidates) {
    if (Test-Path (Join-Path $c 'CMakeLists.txt')) { return $c }
  }
  return $null
}

function Show-Version([string]$label,[string]$path,[string]$ExpectRef,[string]$ExpectTag) {
  if (-not (Test-Cmd 'git')) { Write-Host "[$label] $path (git not available)"; return }
  if (-not (Test-Path (Join-Path $path '.git'))) { Write-Host "[$label] $path (not a git repo)"; return }
  $head = ''
  $tag = ''
  try { $head = (git -C $path rev-parse --short HEAD) } catch {}
  try { $tag = (git -C $path describe --tags --exact-match) } catch {}
  $have = if ($tag) { $tag } else { $head }
  $note = ''
  if ($ExpectTag) {
    if ($tag -and $tag -eq $ExpectTag) { $note = "(matches $ExpectTag)" } else { $note = "(want $ExpectTag, have $have)" }
  } elseif ($ExpectRef) {
    if ($head -and $head.StartsWith($ExpectRef)) { $note = "(matches $ExpectRef)" } else { $note = "(want $ExpectRef, have $have)" }
  }
  Write-Host "[$label] $path ref=$have $note"
}

function Get-Generator([string]$arch,[string]$compiler) {
  $vsDefault = 'Visual Studio 17 2022'
  $msvcArch = switch ($arch) {
    'x86_64' { 'x64' }
    'x86_32' { 'Win32' }
    'arm64'  { 'ARM64' }
    'arm32'  { 'ARM' }
    default  { 'x64' }
  }
  switch ($compiler) {
    'msvc' {
      return @{ G=$vsDefault; A=$msvcArch; Mode='msvc' }
    }
    'mingw' {
      return @{ G='MinGW Makefiles'; A=$null; Mode='mingw' }
    }
    'ninja' {
      if (-not (Test-Cmd 'ninja')) { throw "Requested Ninja generator but 'ninja' command is not available." }
      return @{ G='Ninja'; A=$null; Mode='ninja' }
    }
    'auto' {
      $envGen = $env:CMAKE_GENERATOR
      if ($envGen) {
        $genTrim = $envGen.Trim()
        $genLower = $genTrim.ToLowerInvariant()
        if ($genLower -like 'mingw*') { return @{ G=$genTrim; A=$null; Mode='mingw' } }
        if ($genLower -like 'ninja*') {
          if (-not (Test-Cmd 'ninja')) { Write-Warning "CMAKE_GENERATOR requests Ninja but 'ninja' command not found. Falling back to MSVC." }
          else { return @{ G=$genTrim; A=$null; Mode='ninja' } }
        }
        if ($genLower -like 'visual studio*') { return @{ G=$genTrim; A=$msvcArch; Mode='msvc' } }
      }
      if (Test-Cmd 'ninja') { return @{ G='Ninja'; A=$null; Mode='ninja' } }
      return @{ G=$vsDefault; A=$msvcArch; Mode='msvc' }
    }
    default {
      throw "Unhandled compiler mode '$compiler'"
    }
  }
}

function Invoke-CMakeConfigure([string]$src,[string]$bdir,[hashtable]$gen,[string[]]$defs) {
  New-Item -ItemType Directory -Force -Path $bdir | Out-Null
  $args = @('-S', $src, '-B', $bdir, '-D', 'BUILD_SHARED_LIBS=OFF', '-D', 'CMAKE_POSITION_INDEPENDENT_CODE=ON', '-D', 'CMAKE_BUILD_TYPE=Release')
  if ($gen.G) { $args += @('-G', $gen.G) }
  if ($gen.A) { $args += @('-A', $gen.A) }
  foreach ($d in $defs) { $args += $d }
  & cmake @args
}

function Invoke-CMakeBuild([string]$bdir,[string[]]$targets) {
  $args = @('--build', $bdir, '--config', 'Release')
  if ($Jobs -gt 0) { $args += @('--parallel', "$Jobs") }
  if ($targets -and $targets.Count -gt 0) { $args += @('--target'); $args += $targets }
  & cmake @args
}

function Copy-Binary([string]$bdir,[string]$tgt,[string]$outdir) {
  New-Item -ItemType Directory -Force -Path $outdir | Out-Null
  $exe = "$tgt.exe"
  $candidates = @()
  $candidates += (Join-Path $bdir $exe)
  $candidates += (Join-Path (Join-Path $bdir 'bin') $exe)
  $candidates += (Join-Path (Join-Path $bdir 'Release') $exe)
  $candidates += (Join-Path (Join-Path (Join-Path $bdir 'bin') 'Release') $exe)
  $candidates += (Join-Path $bdir $tgt)
  $candidates += (Join-Path (Join-Path $bdir 'bin') $tgt)
  foreach ($c in $candidates) {
    if (Test-Path $c) { Copy-Item $c -Destination $outdir -Force; Write-Host "Copied $(Split-Path $c -Leaf) -> $outdir"; return }
  }
  Write-Warning "Could not locate built binary '$tgt' under $bdir"
}

function Get-ProjectOutDir([string]$arch,[string]$device,[string]$project) {
  $osSegment = if ($script:OutOsId) { $script:OutOsId } else { $OS_ID }
  return Join-Path (Join-Path (Join-Path (Join-Path $OUT $arch) $osSegment) $device) $project
}

function Resolve-MingwRuntimePaths {
  $candidates = New-Object System.Collections.Generic.List[string]
  $pathParts = ($env:PATH -split ';') | Where-Object { $_ -and $_.Trim() -ne '' }
  foreach ($p in $pathParts) { [void]$candidates.Add($p.Trim()) }
  foreach ($var in @('MINGW_HOME','MINGW64_HOME','MSYS2_HOME','MSYSTEM_PREFIX')) {
    $val = [Environment]::GetEnvironmentVariable($var)
    if ($val -and (Test-Path $val)) {
      [void]$candidates.Add($val)
      $bin = Join-Path $val 'bin'
      if (Test-Path $bin) { [void]$candidates.Add($bin) }
    }
  }
  foreach ($extra in @('C:\mingw64\bin','C:\msys64\mingw64\bin')) {
    if (Test-Path $extra) { [void]$candidates.Add($extra) }
  }
  $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
  $result = New-Object System.Collections.Generic.List[string]
  foreach ($cand in $candidates) {
    $trim = $cand.Trim()
    if ($trim -eq '') { continue }
    if (-not (Test-Path $trim)) { continue }
    if ($seen.Add($trim)) { [void]$result.Add($trim) }
  }
  return $result.ToArray()
}

function Ensure-MingwRuntime([string]$targetDir) {
  if ($script:CompilerMode -ne 'mingw') { return }
  if (-not $targetDir) { return }
  if (-not (Test-Path $targetDir)) { return }
  if ($script:RuntimeDirsCopied.Contains($targetDir)) { return }
  if (-not $script:MingwRuntimePaths -or $script:MingwRuntimePaths.Count -eq 0) {
    $script:MingwRuntimePaths = Resolve-MingwRuntimePaths
  }
  foreach ($dll in $script:MingwRuntimeDlls) {
    $source = $null
    foreach ($dir in $script:MingwRuntimePaths) {
      $candidate = Join-Path $dir $dll
      if (Test-Path $candidate) { $source = $candidate; break }
    }
    if ($source) {
      try {
        Copy-Item $source -Destination (Join-Path $targetDir $dll) -Force
        Write-Host "Copied $dll -> $targetDir"
      } catch {
        Write-Warning "Failed to copy $dll from $source to ${targetDir}: $_"
      }
    } else {
      Write-Warning "MinGW runtime DLL '$dll' not found in PATH; ensure MinGW bin directory is accessible."
    }
  }
  $script:RuntimeDirsCopied.Add($targetDir) | Out-Null
}

function Get-AvailableTargets([string]$bdir) {
  try { return ((& cmake --build $bdir --config Release --target help) -join "`n") } catch { return '' }
}
function Assert-BackendEnabled([string]$bdir,[string]$device,[string]$project) {
  $cache = Join-Path $bdir 'CMakeCache.txt'
  if (-not (Test-Path $cache)) { return }
  $txt = Get-Content -Raw -LiteralPath $cache
  switch ($device) {
    'vulkan' {
      if ($txt -notmatch 'GGML_VULKAN:BOOL=ON|GGML_VULKAN:BOOL=TRUE') {
        Write-Warning "$project [$device]: GGML_VULKAN not enabled by CMake; falling back to CPU. Ensure Vulkan SDK installed and VULKAN_SDK is set."
      } elseif ($txt -notmatch 'Vulkan_FOUND:INTERNAL=1|Vulkan_FOUND:BOOL=TRUE') {
        Write-Warning "$project [$device]: Vulkan not found (Vulkan_FOUND=FALSE); check VULKAN_SDK and build environment."
      }
    }
    'cuda' {
      if ($txt -notmatch 'GGML_CUDA:BOOL=ON|GGML_CUDA:BOOL=TRUE') {
        Write-Warning "$project [$device]: GGML_CUDA not enabled; check CUDA toolkit and CMake logs."
      }
    }
    'opencl' {
      if ($txt -notmatch 'GGML_OPENCL:BOOL=ON|GGML_OPENCL:BOOL=TRUE') {
        Write-Warning "$project [$device]: GGML_OPENCL not enabled; ensure OpenCL headers/libs available."
      }
    }
  }
}


function Build-Llama([string]$device,[string]$arch) {
  $src = Resolve-Src 'llama.cpp' $LlamaSrc 'LLAMA_SRC' @((Join-Path $ROOT 'llama.cpp'), (Join-Path $EXTERN 'llama.cpp'))
  if (-not $src) { throw "llama.cpp source not found. Provide -LlamaSrc or set LLAMA_SRC or place repo at .\llama.cpp or .\external\llama.cpp." }
  Show-Version 'llama.cpp' $src $LLAMA_EXPECT_REF ''
  $bdir = Join-Path (Join-Path $BUILD 'llama.cpp') $device
  if ($Clean) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $bdir }
  $defs = @('-D', 'LLAMA_CURL=OFF', '-D', 'LLAMA_BUILD_TESTS=OFF', '-D', 'LLAMA_BUILD_EXAMPLES=ON', '-D', 'LLAMA_BUILD_SERVER=ON')
  switch ($device) {
    'vulkan' {  $defs += @('-D','GGML_VULKAN=ON','-D','SD_VULKAN=ON') }
    'cuda'   {  $defs += @('-D','GGML_CUDA=ON','-D','SD_CUDA=ON','-D','GGML_NATIVE=OFF') }
    'opencl' {  $defs += @('-D','GGML_OPENCL=ON','-D','SD_OPENCL=ON') }
    default  { $defs += @('-D','GGML_VULKAN=OFF','-D','GGML_CUDA=OFF','-D','GGML_OPENCL=OFF') }
  }
  if ($script:CompilerMode -eq 'mingw') { $defs += @('-D','CMAKE_OBJECT_PATH_MAX=196') }
  $gen = $GeneratorSpec
  Invoke-CMakeConfigure $src $bdir $gen $defs
  if ($device -in @('vulkan','cuda','opencl')) { Assert-BackendEnabled $bdir $device 'llama.cpp' }
  $help = Get-AvailableTargets $bdir
  $targets = @()
  foreach ($t in @('llama-server','llama-quantize')) { if ($help -match [regex]::Escape($t)) { $targets += $t } }
  if ($targets.Count -eq 0) { Invoke-CMakeBuild $bdir @() } else { Invoke-CMakeBuild $bdir $targets }
  $out = Get-ProjectOutDir $arch $device 'llama.cpp'
  Copy-Binary $bdir 'llama-server' $out
  Copy-Binary $bdir 'llama-quantize' $out
  Ensure-MingwRuntime $out
}
function Build-Whisper([string]$device,[string]$arch) {
  $src = Resolve-Src 'whisper.cpp' $WhisperSrc 'WHISPER_SRC' @((Join-Path $ROOT 'whisper.cpp'), (Join-Path $EXTERN 'whisper.cpp'))
  if (-not $src) { throw "whisper.cpp source not found. Provide -WhisperSrc or set WHISPER_SRC or place repo at .\whisper.cpp or .\external\whisper.cpp." }
  Show-Version 'whisper.cpp' $src '' $WHISPER_EXPECT_TAG
  $bdir = Join-Path (Join-Path $BUILD 'whisper.cpp') $device
  if ($Clean) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $bdir }
  $defs = @('-D','WHISPER_BUILD_TESTS=OFF','-D','WHISPER_BUILD_EXAMPLES=ON')
  switch ($device) {
    'vulkan' {  $defs += @('-D','GGML_VULKAN=ON','-D','SD_VULKAN=ON') }
    'cuda'   {  $defs += @('-D','GGML_CUDA=ON','-D','SD_CUDA=ON','-D','GGML_NATIVE=OFF') }
    'opencl' {  $defs += @('-D','GGML_OPENCL=ON','-D','SD_OPENCL=ON') }
    default  { $defs += @('-D','GGML_VULKAN=OFF','-D','GGML_CUDA=OFF','-D','GGML_OPENCL=OFF') }
  }
  $gen = $GeneratorSpec
  Invoke-CMakeConfigure $src $bdir $gen $defs
  if ($device -in @('vulkan','cuda','opencl')) { Assert-BackendEnabled $bdir $device 'whisper.cpp' }
  Invoke-CMakeBuild $bdir @('whisper-cli')
  $out = Get-ProjectOutDir $arch $device 'whisper.cpp'
  Copy-Binary $bdir 'whisper-cli' $out
  Ensure-MingwRuntime $out
}
function Build-SD([string]$device,[string]$arch) {
  $src = Resolve-Src 'stable-diffusion.cpp' $SDSrc 'SD_SRC' @((Join-Path $ROOT 'stable-diffusion.cpp'), (Join-Path $EXTERN 'stable-diffusion.cpp'))
  if (-not $src) { throw "stable-diffusion.cpp source not found. Provide -SDSrc or set SD_SRC or place repo at .\stable-diffusion.cpp or .\external\stable-diffusion.cpp." }
  Show-Version 'stable-diffusion.cpp' $src $SD_EXPECT_REF ''
  $bdir = Join-Path (Join-Path $BUILD 'stable-diffusion.cpp') $device
  if ($Clean) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $bdir }
  $defs = @()
  switch ($device) {
    'vulkan' {  $defs += @('-D','GGML_VULKAN=ON','-D','SD_VULKAN=ON') }
    'cuda'   {  $defs += @('-D','GGML_CUDA=ON','-D','SD_CUDA=ON','-D','GGML_NATIVE=OFF') }
    'opencl' {  $defs += @('-D','GGML_OPENCL=ON','-D','SD_OPENCL=ON') }
    default  { $defs += @('-D','GGML_VULKAN=OFF','-D','GGML_CUDA=OFF','-D','GGML_OPENCL=OFF') }
  }
  $gen = $GeneratorSpec
  Invoke-CMakeConfigure $src $bdir $gen $defs
  if ($device -in @('vulkan','cuda','opencl')) { Assert-BackendEnabled $bdir $device 'stable-diffusion.cpp' }
  Invoke-CMakeBuild $bdir @('sd')
  $out = Get-ProjectOutDir $arch $device 'stable-diffusion.cpp'
  Copy-Binary $bdir 'sd' $out
  Ensure-MingwRuntime $out
}

function Build-TTS([string]$device,[string]$arch) {
  $project = 'tts.cpp'
  $cpuOutDir = Get-ProjectOutDir $arch 'cpu' $project
  $binaryName = 'tts-cli.exe'
  if ($device -ne 'cpu') {
    $targetDir = Get-ProjectOutDir $arch $device $project
    $targetBinary = Join-Path $targetDir $binaryName
    if (Test-Path $targetBinary) {
      Write-Host "[info] tts.cpp: reusing CPU binary for device '$device'"
      return
    }
    $cpuBinary = Join-Path $cpuOutDir $binaryName
    if (-not (Test-Path $cpuBinary)) {
      Write-Host "[info] tts.cpp: CPU binary missing; building CPU variant first."
      Build-TTS 'cpu' $arch
      if (Test-Path $targetBinary) {
        Write-Host "[info] tts.cpp: populated device '$device' from CPU build"
        return
      }
    }
    if (-not (Test-Path $cpuBinary)) {
      Write-Warning "tts.cpp: CPU binary unavailable; cannot populate device '$device'"
      return
    }
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Copy-Item $cpuBinary -Destination $targetDir -Force
    Write-Host "Copied $(Split-Path $cpuBinary -Leaf) -> $targetDir"
    Ensure-MingwRuntime $targetDir
    return
  }
  $src = Resolve-Src 'tts.cpp' $TTSSrc 'TTS_SRC' @((Join-Path $ROOT 'tts.cpp'), (Join-Path $EXTERN 'tts.cpp'))
  if (-not $src) { throw "tts.cpp source not found. Provide -TTSSrc or set TTS_SRC or place repo at .\tts.cpp or .\external\tts.cpp." }
  Show-Version 'tts.cpp' $src $TTS_EXPECT_REF ''
  $bdir = Join-Path (Join-Path $BUILD 'tts.cpp') $device
  if ($Clean) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $bdir }
  $defs = @(
    '-D','TTS_BUILD_EXAMPLES=ON',
    '-D','GGML_VULKAN=OFF',
    '-D','GGML_CUDA=OFF',
    '-D','GGML_OPENCL=OFF'
  )
  $gen = $GeneratorSpec
  Invoke-CMakeConfigure $src $bdir $gen $defs
  if ($device -in @('vulkan','cuda','opencl')) { Assert-BackendEnabled $bdir $device 'tts.cpp' }
  Invoke-CMakeBuild $bdir @('tts-cli')
  $out = $cpuOutDir
  Copy-Binary $bdir 'tts-cli' $out
  $cpuBinary = Join-Path $out $binaryName
  Ensure-MingwRuntime $out
  if (Test-Path $cpuBinary -and $script:AllDevices -and $script:AllDevices.Count -gt 0) {
    foreach ($extra in $script:AllDevices) {
      if ($extra -eq 'cpu') { continue }
      $extraDir = Get-ProjectOutDir $arch $extra $project
      New-Item -ItemType Directory -Force -Path $extraDir | Out-Null
      Copy-Item $cpuBinary -Destination $extraDir -Force
      Write-Host "Copied $(Split-Path $cpuBinary -Leaf) -> $extraDir"
      Ensure-MingwRuntime $extraDir
    }
  }
}
$arch = Resolve-Arch
$CompilerMode = Resolve-CompilerMode $CompilerRequest
$GeneratorSpec = Get-Generator $arch $CompilerMode
$CompilerMode = if ($GeneratorSpec.ContainsKey('Mode')) { $GeneratorSpec.Mode } else { $CompilerMode }
$script:CompilerMode = $CompilerMode
$buildOsTag = if ($CompilerMode -eq 'mingw') { 'win7' } else { $OS_ID }
$OutOsId = $buildOsTag
$script:OutOsId = $OutOsId
$BUILD = Join-Path $ROOT ("build-$arch-$buildOsTag")
$devs = Resolve-Devices $Devices
$devs = Filter-DevicesForCompiler $devs $CompilerMode
$script:AllDevices = $devs
if ($Projects -eq 'all') { $projs = @('llama','whisper','sd','tts') } else { $projs = $Projects.Split(',') }
Write-Host "==> OS=$OS_ID ARCH=$arch COMPILER=$CompilerMode OUT_OS=$OutOsId GENERATOR=$($GeneratorSpec.G) BUILD_DIR=$BUILD DEVICES=$($devs -join ',') PROJECTS=$($projs -join ',')"

foreach ($d in $devs) {
  foreach ($p in $projs) {
    Write-Host "--- Building $p [$d] ---"
    switch ($p.Trim()) {
      'llama'   { Build-Llama   $d $arch }
      'whisper' { Build-Whisper $d $arch }
      'sd'      { Build-SD      $d $arch }
      'tts'     { Build-TTS     $d $arch }
      default   { throw "Unknown project: $p" }
    }
  }
}

Write-Host "Done. Artifacts under: $OUT\$arch\$OutOsId"






