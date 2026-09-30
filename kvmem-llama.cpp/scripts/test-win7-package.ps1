$ErrorActionPreference = 'Stop'
$validator = Join-Path $PSScriptRoot 'verify-win7-package.ps1'
$temp = Join-Path ([System.IO.Path]::GetTempPath()) ('win7-package-test-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $temp | Out-Null

# Mock dumpbin so these policy tests need neither MSVC nor real PE binaries.
function dumpbin {
  param($Nologo, $Mode, $File)
  $global:LASTEXITCODE = $global:Win7TestDumpbinExitCode
  if ($Mode -eq '/dependents') {
    foreach ($dep in $global:Win7TestDependencies[(Split-Path $File -Leaf)]) { "    $dep" }
  } else {
    $global:Win7TestImports
  }
}

function Check-Package([string]$Name, [string]$ExpectedError = '') {
  $failure = ''
  try { & $validator -PackageDir $temp } catch { $failure = $_.Exception.Message }
  if ($ExpectedError) {
    if (-not $failure.Contains($ExpectedError)) {
      throw "${Name}: expected '$ExpectedError', got '$failure'"
    }
  } elseif ($failure) {
    throw "${Name}: unexpected failure: $failure"
  }
  Write-Host "PASS: $Name"
}

try {
  New-Item -ItemType File -Path (Join-Path $temp 'server.exe'), (Join-Path $temp 'llama.dll') | Out-Null
  $global:Win7TestDumpbinExitCode = 0
  $global:Win7TestImports = '    123 GetSystemTimeAsFileTime'
  $global:Win7TestDependencies = @{
    'server.exe' = @('llama.dll', 'KERNEL32.dll')
    'llama.dll' = @('KERNEL32.dll', 'MSVCP140.dll', 'VCOMP140.DLL', 'api-ms-win-crt-runtime-l1-1-0.dll')
  }
  Check-Package 'shared CRT prerequisites and case-insensitive names'

  $global:Win7TestDependencies['llama.dll'] += 'cudart64_110.dll'
  Check-Package 'unbundled dynamic cudart is still forbidden' 'llama.dll -> cudart64_110.dll'
  New-Item -ItemType File -Path (Join-Path $temp 'cudart64_110.dll') | Out-Null
  $global:Win7TestDependencies['cudart64_110.dll'] = @('KERNEL32.dll')
  Check-Package 'bundling cudart does not repair linkage' 'Unexpected dynamic cudart'
  Remove-Item (Join-Path $temp 'cudart64_110.dll')

  $global:Win7TestDependencies['llama.dll'] = @('KERNEL32.dll', 'missing-backend.dll')
  Check-Package 'missing transitive dependency' 'missing-backend.dll (missing packaged dependency)'
  $global:Win7TestDependencies['llama.dll'] = @('api-ms-win-core-libraryloader-l1-2-0.dll')
  Check-Package 'Windows 8 API set' 'api-ms-win-core-libraryloader-l1-2-0.dll (not allowed'
  $global:Win7TestDependencies['llama.dll'] = @('KERNEL32.dll')
  $global:Win7TestImports = '    456 WaitOnAddress'
  Check-Package 'new API imported from kernel32 directly' 'Windows 8+ function WaitOnAddress'
  $global:Win7TestImports = ''

  $global:Win7TestDumpbinExitCode = 1
  Check-Package 'dumpbin failure cannot silently pass' 'dumpbin failed'
  $global:Win7TestDumpbinExitCode = 0
  $global:Win7TestDependencies['llama.dll'] = @()
  Check-Package 'empty dumpbin output cannot silently pass' 'No DLL dependencies reported'
} finally {
  Remove-Item -LiteralPath $temp -Recurse -Force
}
