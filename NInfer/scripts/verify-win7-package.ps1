#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PackageDir,
    # A trusted snapshot of the TARGET's x64 System32, not the build host's DLLs.
    [Parameter(Mandatory = $true)][string]$Win7SystemDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$PackageDir = (Resolve-Path -LiteralPath $PackageDir).Path
$Win7SystemDir = (Resolve-Path -LiteralPath $Win7SystemDir).Path
if (-not (Get-Command dumpbin -ErrorAction SilentlyContinue)) { throw 'dumpbin not found' }

$systemDlls = @(
    'kernel32.dll', 'kernelbase.dll', 'ntdll.dll', 'advapi32.dll', 'bcrypt.dll',
    'crypt32.dll', 'cryptbase.dll', 'user32.dll', 'gdi32.dll', 'shell32.dll',
    'shlwapi.dll', 'ole32.dll', 'oleaut32.dll', 'ws2_32.dll', 'mswsock.dll',
    'iphlpapi.dll', 'dnsapi.dll', 'secur32.dll', 'sspicli.dll', 'version.dll',
    'winmm.dll', 'psapi.dll', 'dbghelp.dll', 'powrprof.dll', 'setupapi.dll',
    'cfgmgr32.dll', 'rpcrt4.dll', 'normaliz.dll', 'msvcrt.dll', 'imm32.dll'
)
$crtPattern = '^(api-ms-win-crt-[a-z0-9-]+|ucrtbase|msvcp140(?:_[a-z0-9]+)?|vcruntime140(?:_1)?|vcomp140|concrt140)\.dll$'
$vcPattern = '^(msvcp140(?:_[a-z0-9]+)?|vcruntime140(?:_1)?|vcomp140|concrt140)\.dll$'
$forbiddenDllPattern = '^(cudart64_.*\.dll|api-ms-win-core-.*\.dll|ext-ms-.*\.dll|nvToolsExt.*\.dll)$'
$newApiPattern = '^(WaitOnAddress|WakeByAddressSingle|WakeByAddressAll|SetThreadDescription|GetThreadDescription|GetSystemTimePreciseAsFileTime|GetCurrentThreadStackLimits|GetOverlappedResultEx|GetTempPath2[AW])$'
$exportsCache = @{}
$headersChecked = @{}
$prerequisites = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

function Read-Dump([string]$Mode, [string]$File) {
    $lines = @(& dumpbin /nologo $Mode $File 2>&1 | ForEach-Object { "$_" })
    if ($LASTEXITCODE -ne 0) { throw "dumpbin $Mode failed ($LASTEXITCODE) for ${File}: $($lines -join '`n')" }
    if ($lines.Count -eq 0) { throw "Empty dumpbin $Mode output for $File" }
    return $lines
}

function Check-Headers([string]$File) {
    if ($headersChecked.ContainsKey($File)) { return }
    $text = (Read-Dump '/headers' $File) -join "`n"
    if ($text -notmatch '(?im)^\s*8664 machine \(x64\)\s*$') { throw "Not an x64 PE: $File" }
    foreach ($field in @('operating system version', 'subsystem version')) {
        if ($text -notmatch "(?im)^\s*(\d+)\.(\d+) $field\s*$") { throw "Missing PE $field in $File" }
        if ([int]$Matches[1] -gt 6 -or ([int]$Matches[1] -eq 6 -and [int]$Matches[2] -gt 1)) {
            throw "PE $field is newer than Windows 7: $File"
        }
    }
    $headersChecked[$File] = $true
}

$baselineKernel = Join-Path $Win7SystemDir 'kernel32.dll'
if (-not (Test-Path -LiteralPath $baselineKernel -PathType Leaf)) { throw 'Win7SystemDir must contain the target kernel32.dll' }
$version = (Get-Item -LiteralPath $baselineKernel).VersionInfo
if ($version.FileMajorPart -ne 6 -or $version.FileMinorPart -ne 1 -or $version.FileBuildPart -ne 7601) {
    throw 'Win7SystemDir is not a Windows 7 SP1 (6.1.7601) baseline; never use the build host System32'
}
Check-Headers $baselineKernel

function Resolve-Dependency([string]$Name, [string]$Importer) {
    if ($Name -match $forbiddenDllPattern) { throw "$Importer -> $Name (forbidden Win7 dependency)" }
    if ($Name -in $systemDlls -or $Name -match $crtPattern -or $Name -ieq 'nvcuda.dll') {
        $path = Join-Path $Win7SystemDir $Name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "$Importer -> $Name (missing target prerequisite in Win7SystemDir)"
        }
        if ($Name -match $vcPattern) {
            $v = (Get-Item -LiteralPath $path).VersionInfo
            if ($v.FileMajorPart -ne 14 -or $v.FileMinorPart -ne 29) {
                throw "$Name requires the controlled VC++ 2019 14.29 x64 runtime, got $($v.FileVersion)"
            }
        }
        [void]$prerequisites.Add($Name)
    } else {
        $path = Join-Path $PackageDir $Name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "$Importer -> $Name (missing packaged dependency)"
        }
    }
    Check-Headers $path
    return $path
}

function Read-Exports([string]$File) {
    if (-not $exportsCache.ContainsKey($File)) {
        $symbols = [System.Collections.Generic.Dictionary[string,string]]::new([System.StringComparer]::Ordinal)
        foreach ($line in (Read-Dump '/exports' $File)) {
            # Named exports, including forwarded exports without an RVA.
            if ($line -match '^\s+(\d+)\s+[0-9A-Fa-f]+\s+(?:[0-9A-Fa-f]{8,16}\s+)?([^\s=]+)(?:\s+=\s+\S+)?(?:\s+\(forwarded to ([^)]+)\))?\s*$') {
                $ordinal = '#' + $Matches[1]
                $name = $Matches[2]
                $forward = if ($Matches.ContainsKey(3)) { $Matches[3] } else { '' }
                $symbols[$ordinal] = $forward
                if ($name -ne '[NONAME]') { $symbols[$name] = $forward }
            } elseif ($line -match '^\s+(\d+)\s+[0-9A-Fa-f]{8,16}\s+\[NONAME\]\s*$') {
                $symbols['#' + $Matches[1]] = ''
            }
        }
        if ($symbols.Count -eq 0) { throw "No parseable exports in $File" }
        $exportsCache[$File] = $symbols
    }
    # Do not let PowerShell enumerate an export table into the pipeline.
    return ,$exportsCache[$File]
}

function Check-Import([string]$Dll, [string]$Symbol, [string]$Importer, [System.Collections.Generic.HashSet[string]]$Chain) {
    $path = Resolve-Dependency $Dll $Importer
    $key = $path + '!' + $Symbol
    if (-not $Chain.Add($key)) { throw "Export forwarder cycle: $key" }
    $symbols = Read-Exports $path
    if (-not $symbols.ContainsKey($Symbol)) { throw "$Importer -> $Dll!$Symbol (API absent from target export table)" }
    $forward = $symbols[$Symbol]
    if ($forward) {
        $dot = $forward.LastIndexOf('.')
        if ($dot -lt 1) { throw "Unparseable export forwarder: $forward" }
        $module = $forward.Substring(0, $dot)
        if (-not $module.EndsWith('.dll', [System.StringComparison]::OrdinalIgnoreCase)) { $module += '.dll' }
        Check-Import $module ($forward.Substring($dot + 1)) $key $Chain
    }
}

$items = @(Get-ChildItem -LiteralPath $PackageDir -Recurse -Force)
foreach ($item in $items) {
    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Package contains a reparse point: $($item.FullName)" }
}
$files = @($items | Where-Object { -not $_.PSIsContainer -and $_.Extension -in @('.exe', '.dll') })
if (-not (Test-Path -LiteralPath (Join-Path $PackageDir 'ninfer.exe') -PathType Leaf)) { throw 'Missing ninfer.exe' }
foreach ($file in $files) {
    # The shipped loader layout is flat. A nested DLL is not made loadable merely
    # by finding its basename somewhere beneath the package directory.
    if ($file.DirectoryName -ine $PackageDir) { throw "PE files must be in the package root: $($file.FullName)" }
    if ($file.Extension -ieq '.exe' -and $file.Name -ine 'ninfer.exe') { throw "Unexpected executable: $($file.Name)" }
    if ($file.Name -match $forbiddenDllPattern -or $file.Name -ieq 'nvcuda.dll' -or
        $file.Name -in $systemDlls -or $file.Name -match $crtPattern) {
        throw "Do not bundle driver, system, CRT or dynamic cudart DLLs: $($file.Name)"
    }
    Check-Headers $file.FullName
    $dependencies = @((Read-Dump '/dependents' $file.FullName) | ForEach-Object {
        if ($_ -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') { $Matches[1] }
    } | Select-Object -Unique)
    if ($dependencies.Count -eq 0) { throw "No DLL dependencies reported for $($file.Name)" }
    foreach ($dep in $dependencies) { [void](Resolve-Dependency $dep $file.Name) }

    $currentDll = ''
    $expectedImports = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($dep in $dependencies) { [void]$expectedImports.Add($dep) }
    $seenImports = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($line in (Read-Dump '/imports' $file.FullName)) {
        if ($line -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') {
            $currentDll = $Matches[1]
            [void]$expectedImports.Add($currentDll)
            if ($currentDll -notin $dependencies) {
                # Also check delay-loaded imports even if /dependents omitted them.
                [void](Resolve-Dependency $currentDll $file.Name)
            }
            continue
        }
        if ($line -match '^\s*(Summary|Section contains)\b') { $currentDll = ''; continue }
        if (-not $currentDll) { continue }
        $symbol = ''
        if ($line -match '^\s+(?:[0-9A-Fa-f]+\s+)?Ordinal\s+(\d+)\s*$') {
            $symbol = '#' + $Matches[1]
        } elseif ($line -match '^\s+[0-9A-Fa-f]+\s+([A-Za-z_?@$][^\s()]*)\s*$') {
            $symbol = $Matches[1]
        }
        if (-not $symbol) { continue }
        if ($symbol -match $newApiPattern) { throw "$($file.Name) imports Windows 8+ API $symbol" }
        $chain = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
        Check-Import $currentDll $symbol $file.Name $chain
        [void]$seenImports.Add($currentDll)
    }
    foreach ($dep in $expectedImports) {
        if (-not $seenImports.Contains($dep)) { throw "No parseable imports for $($file.Name) -> $dep" }
    }
}

Write-Host "Win7 PE/dependency/API checks passed for $($files.Count) files."
Write-Host ('Target prerequisites checked against baseline: ' + (($prerequisites | Sort-Object) -join ', '))
Write-Host 'This is an import check, not a native Win7 startup, GPU or numerical qualification.'
