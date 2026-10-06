#requires -Version 5.1
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$PackageDir,
  [Parameter(Mandatory=$true)][string]$SourceDir,
  [Parameter(Mandatory=$true)][string]$LlamaZip
)

$ErrorActionPreference = 'Stop'
$PackageDir = (Resolve-Path $PackageDir).Path
$SourceDir = (Resolve-Path $SourceDir).Path
$llamaZipPath = (Resolve-Path $LlamaZip).Path

$pyZip = Join-Path ([IO.Path]::GetTempPath()) 'python-3.8.10-embed-amd64.zip'
if (-not (Test-Path $pyZip)) {
  Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.8.10/python-3.8.10-embed-amd64.zip' -OutFile $pyZip
}
$pyStage = Join-Path ([IO.Path]::GetTempPath()) ('strata-python-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $pyStage | Out-Null
try {
  Expand-Archive -LiteralPath $pyZip -DestinationPath $pyStage -Force
  New-Item -ItemType Directory -Force -Path (Join-Path $PackageDir 'python') | Out-Null
  # `-LiteralPath` does not expand the wildcard; use `-Path` so the
  # embeddable distribution's files are copied into the package.
  Copy-Item -Path (Join-Path $pyStage '*') -Destination (Join-Path $PackageDir 'python') -Recurse -Force
} finally {
  if (Test-Path $pyStage) { Remove-Item -LiteralPath $pyStage -Recurse -Force }
}

$site = Join-Path $PackageDir 'python\Lib\site-packages'
New-Item -ItemType Directory -Force -Path $site | Out-Null
$runtimeReq = @('numpy==1.24.4','jinja2==3.1.4','regex==2024.11.6','pyyaml==6.0.2',
  'tqdm==4.66.5','requests==2.32.3','pillow==10.4.0','psutil==6.1.1',
  'markupsafe==2.1.5','certifi==2024.8.30','charset-normalizer==3.3.2',
  'idna==3.7','urllib3==2.2.3','colorama')
$reqFile = Join-Path ([IO.Path]::GetTempPath()) 'strata-runtime-win7.txt'
$runtimeReq | Set-Content -LiteralPath $reqFile -Encoding ASCII
python -m pip install --disable-pip-version-check --no-cache-dir --target $site -r $reqFile
if ($LASTEXITCODE -ne 0) { throw "offline Python dependency install failed: $LASTEXITCODE" }

$pth = Get-ChildItem -LiteralPath (Join-Path $PackageDir 'python') -Filter '*._pth' -File | Select-Object -First 1
if (-not $pth) { throw 'embedded Python _pth file not found' }
$pthLines = @(Get-Content -LiteralPath $pth.FullName)
if ($pthLines -notcontains 'Lib\site-packages') { $pthLines += 'Lib\site-packages' }
if ($pthLines -notcontains 'import site') { $pthLines += 'import site' }
Set-Content -LiteralPath $pth.FullName -Value $pthLines -Encoding ASCII

Copy-Item -LiteralPath (Join-Path $SourceDir 'setup.py') -Destination (Join-Path $PackageDir 'setup.py') -Force
Copy-Item -LiteralPath (Join-Path $SourceDir 'requirements.txt') -Destination (Join-Path $PackageDir 'requirements.txt') -Force
Copy-Item -LiteralPath (Join-Path $SourceDir 'serve') -Destination (Join-Path $PackageDir 'serve') -Recurse -Force
New-Item -ItemType Directory -Force -Path (Join-Path $PackageDir 'tools') | Out-Null
Copy-Item -LiteralPath (Join-Path $SourceDir 'tools\strata_mcp.py') -Destination (Join-Path $PackageDir 'tools\strata_mcp.py') -Force
$thirdParty = Join-Path $PackageDir 'third_party'
New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null
$llamaDest = Join-Path $thirdParty ([IO.Path]::GetFileName($llamaZipPath))
Copy-Item -LiteralPath $llamaZipPath -Destination $llamaDest -Force
Set-Content -LiteralPath ($llamaDest + '.done') -Value 'offline-bundled' -Encoding ASCII

# Match Strata's normal layout: setup.py resolves the engine from engine/.
$engine = Join-Path $PackageDir 'engine'
New-Item -ItemType Directory -Force -Path $engine | Out-Null
foreach ($name in @('strata.exe','strata-device.exe','strata-vision.exe','BUILD.json','cublas64_*.dll','cublasLt64_*.dll','cudart64_*.dll')) {
  Get-ChildItem -LiteralPath $PackageDir -File -Filter $name -ErrorAction SilentlyContinue |
    Move-Item -Destination $engine -Force
}
Set-Content -LiteralPath (Join-Path $PackageDir 'python\.strata-pip.json') -Value (@($runtimeReq) | ConvertTo-Json -Compress) -Encoding UTF8

$modelDir = Join-Path $PackageDir 'models\qwen3.8-flash-next\IQ3_XXS'
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
@('Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf',
   'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf') |
  Set-Content -LiteralPath (Join-Path $modelDir 'PLACE-MODEL-FILES-HERE.txt') -Encoding UTF8

@('@echo off','setlocal','set STRATA_OFFLINE=1','set STRATA_PORTABLE=1','cd /d "%~dp0"','python\python.exe setup.py %*','if errorlevel 1 pause') |
  Set-Content -LiteralPath (Join-Path $PackageDir 'START-HERE.bat') -Encoding ASCII
@('@echo off','setlocal','set STRATA_OFFLINE=1','set STRATA_PORTABLE=1','cd /d "%~dp0"',
  'if not exist "models\qwen3.8-flash-next\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf" (',
  '  echo Put both IQ3_XXS GGUF shards into models\qwen3.8-flash-next\IQ3_XXS first.','  pause','  exit /b 1',')',
  'python\python.exe setup.py --yes --family qwen --model IQ3_XXS --gguf-dir "models\qwen3.8-flash-next\IQ3_XXS" --context 32768 --vision no --no-start',
  'if errorlevel 1 pause ^& exit /b 1','python\python.exe setup.py','if errorlevel 1 pause') |
  Set-Content -LiteralPath (Join-Path $PackageDir 'RUN-IQ3-XXS.bat') -Encoding ASCII
@('Strata Win7 CUDA 11.7 offline package','','Put both IQ3_XXS GGUF shards into models\qwen3.8-flash-next\IQ3_XXS\',
  'Then double-click RUN-IQ3-XXS.bat and open http://127.0.0.1:8080.',
  'Python 3.8.10, service code, engine, strata-vision.exe, cuBLAS DLLs and llama.cpp archive are bundled. The NVIDIA driver supplies nvcuda.dll.') |
  Set-Content -LiteralPath (Join-Path $PackageDir 'README-OFFLINE-WIN7.txt') -Encoding UTF8

Write-Host "Prepared offline Strata package: $PackageDir"
