#requires -Version 5.1
# Behavioral verifier tests. No compiler, GPU, real PE file or Pester is needed.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$validator = Join-Path $PSScriptRoot 'verify-win7-package.ps1'
$temp = Join-Path $PSScriptRoot ('.win7-package-test-' + [guid]::NewGuid().ToString('N'))
$package = Join-Path $temp 'package with spaces'
$system = Join-Path $temp 'Win7 System32'
$script:state = @{}
$passed = 0
$previousExitCode = Get-Variable -Name LASTEXITCODE -Scope Global -ErrorAction SilentlyContinue
$previousExitValue = if ($previousExitCode) { $previousExitCode.Value } else { $null }

# Parse without executing the build or smoke scripts.
foreach ($name in @('build-win7-cuda.ps1', 'verify-win7-package.ps1', 'smoke-win7-package.ps1')) {
    $tokens = $null; $parseErrors = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot $name), [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -gt 0) { throw ("Syntax errors in ${name}: " + ($parseErrors -join '; ')) }
    Write-Host "PASS: PowerShell parser: $name"
}

function New-FixtureFile([string]$Directory, [string]$Name) {
    New-Item -ItemType File -Path (Join-Path $Directory $Name) -Force | Out-Null
}
function Reset-Fixture {
    foreach ($dir in @($package, $system)) {
        if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force }
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    foreach ($name in @('ninfer.exe', 'inference-kernels.dll')) { New-FixtureFile $package $name }
    foreach ($name in @('kernel32.dll', 'kernelbase.dll', 'vcruntime140.dll', 'msvcp140.dll', 'api-ms-win-crt-runtime-l1-1-0.dll', 'nvcuda.dll')) {
        New-FixtureFile $system $name
    }
    $script:state = @{
        FailMode = ''; EmptyMode = ''; Headers = @{}; VcMinor = 29; KernelBuild = 7601
        Dependencies = @{
            'ninfer.exe' = @('KERNEL32.dll', 'inference-kernels.dll')
            'inference-kernels.dll' = @('VCRUNTIME140.dll', 'MSVCP140.dll', 'api-ms-win-crt-runtime-l1-1-0.dll', 'nvcuda.dll')
        }
        Imports = @{
            'ninfer.exe' = @('    KERNEL32.dll', '              1 CloseHandle', '    inference-kernels.dll', '              1 RunInference')
            'inference-kernels.dll' = @(
                '    VCRUNTIME140.dll', '              1 memcpy', '    MSVCP140.dll', '              2 ?runtime@@YAXXZ',
                '    api-ms-win-crt-runtime-l1-1-0.dll', '              3 exit', '    nvcuda.dll', '              4 cuInit'
            )
        }
        Exports = @{
            'kernel32.dll' = @('          1    0          CloseHandle (forwarded to KERNELBASE.CloseHandle)')
            'kernelbase.dll' = @('          1    0 00001000 CloseHandle')
            'inference-kernels.dll' = @('          1    0 00001000 RunInference')
            'vcruntime140.dll' = @('          1    0 00001000 memcpy')
            'msvcp140.dll' = @('          2    1 00001000 ?runtime@@YAXXZ')
            'api-ms-win-crt-runtime-l1-1-0.dll' = @('          3    2 00001000 exit')
            'nvcuda.dll' = @('          4    3 00001000 cuInit')
        }
    }
}

# Mock the actual native-tool boundary. The validator still parses the tool's
# output and resolves real directory entries; no source-text assertions are used.
function dumpbin {
    param($Nologo, $Mode, $File)
    $global:LASTEXITCODE = 0
    if ($Mode -eq $script:state.FailMode) { $global:LASTEXITCODE = 23; return 'dumpbin fixture failure' }
    if ($Mode -eq $script:state.EmptyMode) { return }
    $name = Split-Path $File -Leaf
    switch ($Mode) {
        '/headers' {
            if ($script:state.Headers.ContainsKey($name)) { return $script:state.Headers[$name] }
            return @('             8664 machine (x64)', '            6.00 operating system version', '            6.00 subsystem version')
        }
        '/dependents' {
            '  Image has the following dependencies:'
            foreach ($dep in $script:state.Dependencies[$name]) { "    $dep" }
        }
        '/imports' { return $script:state.Imports[$name] }
        '/exports' {
            '    ordinal hint RVA      name'
            return $script:state.Exports[$name]
        }
        default { throw "Unexpected mock dumpbin mode $Mode" }
    }
}
function Get-Item {
    param([string]$LiteralPath)
    $item = Microsoft.PowerShell.Management\Get-Item -LiteralPath $LiteralPath
    if ($item.DirectoryName -eq $system) {
        $version = if ($item.Name -ieq 'kernel32.dll') {
            [pscustomobject]@{ FileMajorPart = 6; FileMinorPart = 1; FileBuildPart = $script:state.KernelBuild; FileVersion = "6.1.$($script:state.KernelBuild)" }
        } else {
            [pscustomobject]@{ FileMajorPart = 14; FileMinorPart = $script:state.VcMinor; FileBuildPart = 30133; FileVersion = "14.$($script:state.VcMinor).30133" }
        }
        $item | Add-Member -NotePropertyName VersionInfo -NotePropertyValue $version -Force
    }
    return $item
}
function Check-Package([string]$Name, [string]$ExpectedError = '') {
    $failure = ''
    try { & $validator -PackageDir $package -Win7SystemDir $system } catch { $failure = $_.Exception.Message }
    if ($ExpectedError) {
        if (-not $failure.Contains($ExpectedError)) { throw "${Name}: expected '$ExpectedError', got '$failure'" }
    } elseif ($failure) {
        throw "${Name}: unexpected failure: $failure"
    }
    $script:passed++
    Write-Host "PASS: $Name"
}

try {
    Reset-Fixture
    Check-Package 'closed dependencies, forwarded exports, controlled CRT, driver prerequisite, no cuBLAS required'

    $script:state.Dependencies['inference-kernels.dll'] += 'cublas64_11.dll'
    Check-Package 'cuBLAS is required only when actually imported' 'cublas64_11.dll (missing packaged dependency)'
    Reset-Fixture
    $script:state.Dependencies['inference-kernels.dll'] += 'cudart64_110.dll'
    Check-Package 'unbundled dynamic cudart still fails' 'cudart64_110.dll (forbidden Win7 dependency)'

    foreach ($name in @('cudart64_110.dll', 'nvcuda.dll', 'vcruntime140.dll', 'ucrtbase.dll', 'kernel32.dll', 'nvToolsExt64_1.dll')) {
        Reset-Fixture
        New-FixtureFile $package $name
        Check-Package "forbid bundled $name even when unused" 'Do not bundle driver, system, CRT or dynamic cudart DLLs'
    }
    Reset-Fixture
    $script:state.Dependencies['inference-kernels.dll'] += 'missing-transitive.dll'
    Check-Package 'recursive dependency closure' 'missing-transitive.dll (missing packaged dependency)'

    Reset-Fixture
    $script:state.Imports['inference-kernels.dll'] += @('    delay-only.dll', '              1 DelayedWork')
    Check-Package 'delay import omitted by dependents is still resolved' 'delay-only.dll (missing packaged dependency)'
    Reset-Fixture
    $script:state.Imports['inference-kernels.dll'] += '    KERNELBASE.dll'
    Check-Package 'unparsed delay import symbols fail closed' 'No parseable imports'

    Reset-Fixture
    $script:state.Imports['ninfer.exe'] = @('    KERNEL32.dll', '              1 WaitOnAddress')
    Check-Package 'direct newer Windows API' 'Windows 8+ API WaitOnAddress'
    $script:state.Imports['ninfer.exe'] = @('    KERNEL32.dll', '              1 CreateFile2')
    Check-Package 'new API not present in the short denylist' 'CreateFile2 (API absent from target export table)'
    $script:state.Imports['ninfer.exe'] = @('    KERNEL32.dll', '              1 closehandle')
    Check-Package 'export symbol names are case sensitive' 'closehandle (API absent from target export table)'

    Reset-Fixture
    $script:state.Dependencies['ninfer.exe'] += 'api-ms-win-core-synch-l1-2-0.dll'
    Check-Package 'new API set DLL' 'api-ms-win-core-synch-l1-2-0.dll (forbidden Win7 dependency)'
    Reset-Fixture
    $script:state.Exports['kernel32.dll'] = @('          1    0          CloseHandle (forwarded to KERNELBASE.MissingApi)')
    Check-Package 'forwarded API target is checked' 'MissingApi (API absent from target export table)'
    $script:state.Exports['kernelbase.dll'] = @('          1    0          MissingApi (forwarded to KERNEL32.CloseHandle)')
    Check-Package 'forwarder cycles fail closed' 'Export forwarder cycle'

    Reset-Fixture
    $script:state.Imports['ninfer.exe'] = @('    KERNEL32.dll', '        8000000000000001 Ordinal 1', '    inference-kernels.dll', '              1 RunInference')
    Check-Package 'ordinal import resolves against the export table'
    $script:state.Imports['ninfer.exe'][1] = '        8000000000000063 Ordinal 99'
    Check-Package 'unavailable ordinal import fails' '#99 (API absent from target export table)'

    Reset-Fixture
    Remove-Item -LiteralPath (Join-Path $system 'vcruntime140.dll')
    Check-Package 'missing target CRT prerequisite' 'VCRUNTIME140.dll (missing target prerequisite in Win7SystemDir)'
    Reset-Fixture
    $script:state.VcMinor = 44
    Check-Package 'newer installed VC runtime is not silently accepted' 'requires the controlled VC++ 2019 14.29'
    Reset-Fixture
    $script:state.KernelBuild = 19045
    Check-Package 'build host DLLs cannot serve as a Win7 baseline' 'not a Windows 7 SP1'

    Reset-Fixture
    $script:state.Headers['inference-kernels.dll'] = @('              14C machine (x86)', '            6.00 operating system version', '            6.00 subsystem version')
    Check-Package 'transitive DLL architecture' 'Not an x64 PE'
    Reset-Fixture
    $script:state.Headers['ninfer.exe'] = @('             8664 machine (x64)', '            6.00 operating system version', '            6.02 subsystem version')
    Check-Package 'Win8 minimum subsystem' 'subsystem version is newer than Windows 7'
    Reset-Fixture
    $nested = Join-Path $package 'nested'
    New-Item -ItemType Directory -Path $nested | Out-Null
    New-FixtureFile $nested 'hidden.dll'
    Check-Package 'recursive scan rejects DLLs hidden outside the flat loader layout' 'PE files must be in the package root'

    foreach ($mode in @('/headers', '/dependents', '/imports', '/exports')) {
        Reset-Fixture
        $script:state.FailMode = $mode
        Check-Package "native failure propagation: $mode" "dumpbin $mode failed (23)"
        $script:state.FailMode = ''
        $script:state.EmptyMode = $mode
        Check-Package "empty tool output: $mode" "Empty dumpbin $mode output"
    }
    Reset-Fixture
    $script:state.Dependencies['ninfer.exe'] = @()
    Check-Package 'dependency parser cannot succeed without any dependencies' 'No DLL dependencies reported'
    Reset-Fixture
    $script:state.Imports['ninfer.exe'] = @('    KERNEL32.dll', '    inference-kernels.dll')
    Check-Package 'import parser cannot silently skip all symbols' 'No parseable imports'
    Write-Host "All $passed verifier policy tests passed. These mocks do not qualify real PE binaries or a GPU."
} finally {
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Recurse -Force }
    if ($previousExitCode) { $global:LASTEXITCODE = $previousExitValue } else { Remove-Variable -Name LASTEXITCODE -Scope Global -ErrorAction SilentlyContinue }
}
