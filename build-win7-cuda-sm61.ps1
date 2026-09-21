Param(
  [int]$Jobs        = [int]::Parse($env:NUMBER_OF_PROCESSORS),
  [switch]$Clean,
  [string]$LlamaSrc = '',
  [string]$CudaArch = '61',
  [string]$Generator = 'auto',
  [string]$Platform = 'x64'
)

$script = Join-Path $PSScriptRoot 'build-llama-win7-cuda.ps1'
$params = @{
  Jobs = $Jobs
  LlamaSrc = $LlamaSrc
  CudaArch = $CudaArch
  Generator = $Generator
  Platform = $Platform
}
if ($Clean) { $params['Clean'] = $true }
& $script @params
