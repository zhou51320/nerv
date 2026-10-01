#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PackageDir,
    [string]$Model,
    [ValidateRange(30, 3600)][int]$TimeoutSeconds = 600
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$os = [Environment]::OSVersion.Version
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitProcess -or
    $os.Major -ne 6 -or $os.Minor -ne 1 -or $os.Build -ne 7601) {
    throw 'Native smoke requires Windows 7 SP1 x64 and x64 Windows PowerShell 5.1; do not substitute Wine or a newer OS.'
}
$PackageDir = (Resolve-Path -LiteralPath $PackageDir).Path
$exe = Join-Path $PackageDir 'ninfer.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing $exe" }
foreach ($file in (Get-ChildItem -LiteralPath $PackageDir -Recurse -File)) {
    if ($file.Name -match '^(nvcuda|cudart64_.*|nvToolsExt.*|ucrtbase|api-ms-.*|ext-ms-.*|msvcp140.*|vcruntime140.*|vcomp140|concrt140)\.dll$') {
        throw "Forbidden packaged runtime DLL: $($file.FullName). Rebuild into a clean package."
    }
}
$system = [Environment]::SystemDirectory
foreach ($name in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')) {
    $path = Join-Path $system $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Install the VC++ 2019 14.29 x64 Redistributable: missing $name" }
    $v = (Get-Item -LiteralPath $path).VersionInfo
    if ($v.FileMajorPart -ne 14 -or $v.FileMinorPart -ne 29) { throw "$name is not the controlled 14.29 runtime: $($v.FileVersion)" }
}
if (-not (Test-Path -LiteralPath (Join-Path $system 'ucrtbase.dll') -PathType Leaf)) {
    throw 'Install the Win7 UCRT update KB2999226 and prerequisites before running this package.'
}
$driver = Join-Path $system 'nvcuda.dll'
if (-not (Test-Path -LiteralPath $driver -PathType Leaf)) { throw 'Install a Win7 x64 NVIDIA driver; nvcuda.dll must come from the driver, not this package.' }
$driverFile = (Get-Item -LiteralPath $driver).VersionInfo
# NVIDIA's Windows version a.b.1X.YYYY encodes public driver XYY.YY.
$driverRelease = ($driverFile.FileBuildPart % 10) * 10000 + $driverFile.FilePrivatePart
if ($driverRelease -lt 45239) { throw "CUDA 11.x minor-version compatibility requires Windows driver >= 452.39, got $($driverFile.FileVersion)" }

if (-not ('NInferWin7Driver' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class NInferWin7Driver {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true)]
    public static extern IntPtr LoadLibraryExW(string file, IntPtr reserved, uint flags);
    [DllImport("nvcuda.dll")] public static extern int cuInit(uint flags);
    [DllImport("nvcuda.dll")] public static extern int cuDriverGetVersion(out int version);
    [DllImport("nvcuda.dll")] public static extern int cuDeviceGetCount(out int count);
    [DllImport("nvcuda.dll")] public static extern int cuDeviceGet(out int device, int ordinal);
    [DllImport("nvcuda.dll")] public static extern int cuDeviceGetAttribute(out int value, int attribute, int device);
    [DllImport("nvcuda.dll")] public static extern int cuDeviceGetName(StringBuilder name, int length, int device);
    [DllImport("nvcuda.dll")] public static extern int cuDeviceTotalMem_v2(out UIntPtr bytes, int device);
}
'@
}
function Check-Cuda([int]$Code, [string]$Operation) {
    if ($Code -ne 0) { throw "$Operation failed: CUDA driver result $Code" }
}
if ([NInferWin7Driver]::LoadLibraryExW($driver, [IntPtr]::Zero, 8) -eq [IntPtr]::Zero) {
    throw "Cannot load system nvcuda.dll: Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
}
Check-Cuda ([NInferWin7Driver]::cuInit(0)) 'cuInit'
$driverApi = 0
Check-Cuda ([NInferWin7Driver]::cuDriverGetVersion([ref]$driverApi)) 'cuDriverGetVersion'
# Requiring 11070 would wrongly reject Win7's CUDA-11.x-compatible drivers.
# SASS-only 75-real avoids PTX JIT; successful inference still must be tested.
if ($driverApi -lt 11000) { throw "CUDA driver API >= 11.0 required, got $driverApi" }
$count = 0
Check-Cuda ([NInferWin7Driver]::cuDeviceGetCount([ref]$count)) 'cuDeviceGetCount'
if ($count -ne 1) { throw "This smoke targets one visible GPU; found $count. Select the 2080 Ti via CUDA_VISIBLE_DEVICES before starting PowerShell." }
$device = 0
Check-Cuda ([NInferWin7Driver]::cuDeviceGet([ref]$device, 0)) 'cuDeviceGet'
$major = 0; $minor = 0
Check-Cuda ([NInferWin7Driver]::cuDeviceGetAttribute([ref]$major, 75, $device)) 'compute capability major'
Check-Cuda ([NInferWin7Driver]::cuDeviceGetAttribute([ref]$minor, 76, $device)) 'compute capability minor'
$name = [Text.StringBuilder]::new(256)
Check-Cuda ([NInferWin7Driver]::cuDeviceGetName($name, 256, $device)) 'cuDeviceGetName'
$memory = [UIntPtr]::Zero
Check-Cuda ([NInferWin7Driver]::cuDeviceTotalMem_v2([ref]$memory, $device)) 'cuDeviceTotalMem_v2'
if ($major -ne 7 -or $minor -ne 5 -or $name.ToString() -notmatch 'RTX\s+2080\s+Ti') {
    throw "Expected RTX 2080 Ti sm75, got $name sm_$major$minor"
}
if ($memory.ToUInt64() -lt 21GB) { throw "Expected the 22GB-class 2080 Ti (>=21 GiB visible), got $($memory.ToUInt64()) bytes" }
Write-Host ("GPU: {0}; sm75; {1:N2} GiB; NVIDIA {2}.{3:D2}; CUDA driver API {4}" -f
    $name, ($memory.ToUInt64() / 1GB), [int][Math]::Floor($driverRelease / 100), ($driverRelease % 100), $driverApi)

function Quote-WindowsArgument([string]$Value) {
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}
function Run-Cli([string[]]$Arguments, [int]$Seconds) {
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $exe
    $start.WorkingDirectory = $PackageDir
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.StandardOutputEncoding = [Text.Encoding]::UTF8
    $start.StandardErrorEncoding = [Text.Encoding]::UTF8
    $start.Arguments = (($Arguments | ForEach-Object { Quote-WindowsArgument $_ }) -join ' ')
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    try {
        if (-not $process.Start()) { throw 'Could not start ninfer.exe' }
        # Drain both pipes concurrently; model-load diagnostics must not deadlock stdout.
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $timedOut = -not $process.WaitForExit($Seconds * 1000)
        if ($timedOut) { $process.Kill(); $process.WaitForExit() }
        $output = $stdout.GetAwaiter().GetResult()
        $diagnostics = $stderr.GetAwaiter().GetResult()
        Write-Host $output
        if ($diagnostics) { Write-Host $diagnostics }
        if ($timedOut) { throw "ninfer.exe timed out after $Seconds seconds" }
        if ($process.ExitCode -ne 0) { throw "ninfer.exe failed with exit code $($process.ExitCode)" }
        return [pscustomobject]@{ Stdout = $output; Stderr = $diagnostics }
    } finally { $process.Dispose() }
}

$help = Run-Cli @('--help') 60
if ($help.Stdout -notmatch 'usage:' -or $help.Stdout -notmatch 'Win7 baseline:' -or
    $help.Stdout -notmatch 'text-only' -or $help.Stdout -match '--spec\b|--vision\b') {
    throw 'CLI help does not identify the Win7 text-only/no-speculative product'
}
Write-Host 'PASS: native Win7 loader and ninfer --help'
if (-not $Model) {
    Write-Host 'SKIP: model-load/inference (supply -Model with an explicit local .ninfer path). No numerical or performance claim.'
    return
}
$Model = (Resolve-Path -LiteralPath $Model).Path
if ([IO.Path]::GetExtension($Model) -ine '.ninfer') { throw 'Model must be an explicit .ninfer artifact' }
$reader = [IO.BinaryReader]::new([IO.File]::OpenRead($Model))
try {
    $magic = $reader.ReadBytes(8)
    if ([BitConverter]::ToString($magic) -cne '4E-49-4E-46-45-52-00-02') { throw 'Model is not a version-2 .ninfer artifact' }
    $jsonBytes = $reader.ReadUInt64()
    if ($jsonBytes -eq 0 -or $jsonBytes -gt [int]::MaxValue -or $jsonBytes -gt ($reader.BaseStream.Length - 16)) {
        throw 'Invalid .ninfer JSON directory length'
    }
    $metadata = [Text.Encoding]::UTF8.GetString($reader.ReadBytes([int]$jsonBytes)) | ConvertFrom-Json
    if ($metadata.identity.model_id -cne 'qwen3.6-27b' -or $metadata.identity.weights_id -cne 'groupwise-int') {
        throw 'This smoke accepts only qwen3.6-27b/groupwise-int; the Engine performs the full artifact validation'
    }
} finally { $reader.Dispose() }
$result = Run-Cli @(
    $Model, '--prompt', 'Reply with one short sentence about CUDA.', '--no-thinking', '--greedy',
    '--device', '0', '--max-context', '2048', '--prefill-chunk', '256', '--max-new', '32',
    '--kv-dtype', 'int8', '--no-cuda-graph'
) $TimeoutSeconds
if ([string]::IsNullOrWhiteSpace($result.Stdout)) { throw 'Inference succeeded but produced no answer content' }
Write-Host 'PASS: real .ninfer model load and text generation. This smoke is not a numerical oracle or a performance benchmark.'
