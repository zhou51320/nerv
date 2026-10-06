#requires -Version 5.1
[CmdletBinding()]
param(
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$MetaPath = Join-Path $Root 'third_party\strata-upstream.json'
$Meta = Get-Content -LiteralPath $MetaPath -Raw | ConvertFrom-Json
$Dest = Join-Path $Root 'third_party\Strata'
$Archive = Join-Path ([IO.Path]::GetTempPath()) ('strata-' + $Meta.commit + '.zip')

if ((Test-Path (Join-Path $Dest 'CMakeLists.txt')) -and -not $Force) {
  Write-Host "Strata source already exists at $Dest; use -Force only to replace it."
  exit 0
}

Invoke-WebRequest -UseBasicParsing -Uri $Meta.archive_url -OutFile $Archive
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
if ($actual -ne $Meta.archive_sha256.ToLowerInvariant()) {
  throw "Strata archive SHA256 mismatch: expected $($Meta.archive_sha256), got $actual"
}

$stage = Join-Path ([IO.Path]::GetTempPath()) ('strata-stage-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $stage | Out-Null
try {
  Expand-Archive -LiteralPath $Archive -DestinationPath $stage -Force
  $src = Get-ChildItem -LiteralPath $stage -Directory | Select-Object -First 1
  if (-not $src -or -not (Test-Path (Join-Path $src.FullName 'CMakeLists.txt'))) { throw 'Invalid Strata archive layout' }
  if (Test-Path (Join-Path $Dest 'CMakeLists.txt')) {
    $marker = Join-Path $Dest '.nerv-strata-owned'
    if (-not (Test-Path $marker)) { throw "Refusing to replace an unowned Strata directory: $Dest" }
    Remove-Item -LiteralPath $Dest -Recurse -Force
  }
  New-Item -ItemType Directory -Force -Path (Split-Path $Dest -Parent) | Out-Null
  Move-Item -LiteralPath $src.FullName -Destination $Dest
  Set-Content -LiteralPath (Join-Path $Dest '.nerv-strata-owned') -Value $Meta.commit -Encoding ASCII
  Copy-Item -LiteralPath $MetaPath -Destination (Join-Path $Dest 'UPSTREAM.json') -Force
  Write-Host "Fetched Strata $($Meta.commit) -> $Dest"
} finally {
  if (Test-Path $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
}
