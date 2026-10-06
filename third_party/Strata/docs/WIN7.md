# Windows 7 / CUDA 11.x profile

This checkout carries the `STRATA_WIN7` build profile for the project's
existing HTTP/OpenAI/Anthropic/MCP/image service. It does not remove those
features; it only selects a CUDA 11.7/11.8 `sm_75-real` image, static cudart,
Win7 target macros and the repository's YY-Thunks object.

From the repository root, in an x64 MSVC v142 (14.29) developer shell:

```powershell
.\scripts\build-strata-win7-cuda.ps1
```

The script builds `strata.exe` and `strata-device.exe`. The Python service is
started separately with a Win7-compatible Python runtime and its normal
configuration; no service endpoint is disabled by this profile.

By default the final package is written to `artifacts/Strata/` and is suitable
for direct compression/upload. The package contains `BUILD.json` with
`system: win`, `compatibility_label: win7`, and the fixed upstream commit. The
CUDA driver, `nvcuda.dll`, cudart DLLs, and Windows system DLLs are never
copied into the package. As with the llama.cpp Win7 package, only the cuBLAS
runtime DLLs (`cublas64_*.dll` and `cublasLt64_*.dll`) are bundled alongside
the executables; the unmodified CMake build remains in
`build-strata-win7-cuda/`.

`PrefetchVirtualMemory` and the server's power-throttling hint are optional
Win8+/Win10 APIs. They are resolved at runtime and skipped when absent, so a
missing export does not prevent startup on Win7. Native Win7 startup, CUDA
driver loading, Python dependency installation and model inference still need
qualification on the target machine.
