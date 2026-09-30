Param([Parameter(Mandatory = $true)][string]$PackageDir)

$ErrorActionPreference = 'Stop'
if (-not (Get-Command dumpbin -ErrorAction SilentlyContinue)) { throw 'dumpbin not found' }
$files = @(Get-ChildItem -LiteralPath $PackageDir -File | Where-Object { $_.Extension -in @('.exe', '.dll') })
if ($files.Count -eq 0) { throw "No PE files in $PackageDir" }
$names = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
$files | ForEach-Object { [void]$names.Add($_.Name) }

# Windows 7 inbox DLLs, not whatever happens to exist in the build host's System32.
$systemDlls = @(
  'kernel32.dll', 'ntdll.dll', 'advapi32.dll', 'bcrypt.dll', 'crypt32.dll',
  'user32.dll', 'gdi32.dll', 'shell32.dll', 'shlwapi.dll', 'ole32.dll', 'oleaut32.dll',
  'ws2_32.dll', 'mswsock.dll', 'iphlpapi.dll', 'dnsapi.dll', 'secur32.dll',
  'version.dll', 'winmm.dll', 'psapi.dll', 'dbghelp.dll', 'powrprof.dll',
  'setupapi.dll', 'cfgmgr32.dll', 'rpcrt4.dll', 'normaliz.dll', 'msvcrt.dll'
)
# These are explicit target-machine prerequisites; do not copy the runner's CRT.
$crtPattern = '^(api-ms-win-crt-[a-z0-9-]+|ucrtbase|msvcp140(?:_[a-z0-9]+)?|vcruntime140(?:_1)?|vcomp140)\.dll$'
$newApiPattern = '\b(WaitOnAddress|WakeByAddressSingle|WakeByAddressAll|SetThreadDescription|GetThreadDescription|GetSystemTimePreciseAsFileTime|GetCurrentThreadStackLimits|GetOverlappedResultEx|GetTempPath2[AW])\b'
$bad = [System.Collections.Generic.List[string]]::new()
foreach ($file in $files) {
  if ($file.Name -match '^cudart64_.*\.dll$') { $bad.Add("Unexpected dynamic cudart: $($file.Name)") }
  $depsOutput = & dumpbin /nologo /dependents $file.FullName 2>&1
  if ($LASTEXITCODE -ne 0) { throw "dumpbin failed for $($file.Name): $depsOutput" }
  $deps = @($depsOutput | ForEach-Object {
    if ($_ -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') { $Matches[1] }
  })
  if ($deps.Count -eq 0) { throw "No DLL dependencies reported for $($file.Name)" }
  foreach ($dep in $deps) {
    if ($dep -match '^cudart64_.*\.dll$' -or $dep -match '^(api-ms-win-core-|ext-ms-)') {
      $bad.Add("$($file.Name) -> $dep (not allowed in the Win7 package)")
    } elseif (-not $names.Contains($dep) -and $dep -notin $systemDlls -and $dep -notmatch $crtPattern) {
      $bad.Add("$($file.Name) -> $dep (missing packaged dependency)")
    }
  }
  $imports = & dumpbin /nologo /imports $file.FullName 2>&1
  if ($LASTEXITCODE -ne 0) { throw "dumpbin imports failed for $($file.Name): $imports" }
  foreach ($line in $imports) {
    if ($line -match $newApiPattern) { $bad.Add("$($file.Name) imports Windows 8+ function $($Matches[1])") }
  }
  if ($file.Name -match '^(msvcp140|vcruntime140|vcomp140).*\.dll$') {
    $version = $file.VersionInfo
    if ($version.FileMajorPart -ne 14 -or $version.FileMinorPart -gt 29) {
      $bad.Add("$($file.Name) is not a verified VC++ 2019 runtime: $($version.FileVersion)")
    }
  }
}
if ($bad.Count -gt 0) { throw ("Win7 package validation failed:`n" + ($bad -join "`n")) }
$files.Name | Sort-Object | Set-Content (Join-Path $PackageDir 'PACKAGE-MANIFEST.txt')
Write-Host 'Win7 import/dependency checks passed (VC++ 2019 x64 + UCRT required on the target).'
