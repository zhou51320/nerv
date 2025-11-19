@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Simplified Windows 7 CPU-only build script using CMake + MinGW.
REM Builds llama.cpp, whisper.cpp, stable-diffusion.cpp, and tts.cpp sequentially.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
pushd "%ROOT%" >nul

set "BUILD_DIR=%ROOT%\build-win7"
set "OUT_BASE=%ROOT%\EVA_BACKEND"
set "OS_TAG=win7"
set "GENERATOR=MinGW Makefiles"
set "JOBS=%NUMBER_OF_PROCESSORS%"
if "%JOBS%"=="" set "JOBS=1"

set "RUNTIME_DLLS=libatomic-1.dll libgcc_s_seh-1.dll libgfortran-5.dll libgomp-1.dll libquadmath-0.dll libstdc++-6.dll libwinpthread-1.dll"

call :detect_arch
set "OUT_DIR=%OUT_BASE%\%ARCH_ID%\%OS_TAG%\cpu"

call :require_cmd cmake
call :require_cmd gcc
call :require_cmd g++
call :require_one mingw32-make make

call :ensure_dir "%BUILD_DIR%"
call :ensure_dir "%OUT_DIR%"

echo === EVA Win7 CPU build ===
echo Root: %ROOT%
echo Build dir: %BUILD_DIR%
echo Output dir: %OUT_DIR%
echo Arch: %ARCH_ID%
echo CMake generator: %GENERATOR%
echo Parallel jobs: %JOBS%

call :build_llama   || goto fail
call :build_whisper || goto fail
call :build_sd      || goto fail
call :build_tts     || goto fail

echo Done. Artifacts in %OUT_DIR% and subdirectories.
goto end

:fail
echo Build failed.
set "ERRORLEVEL=1"

:end
popd >nul
exit /b %ERRORLEVEL%

:require_cmd
where %~1 >nul 2>nul
if errorlevel 1 (
echo [error] Required command %~1 is missing.
exit /b 1
)
exit /b 0

:require_one
if "%~1"=="" goto require_one_missing
where %~1 >nul 2>nul
if errorlevel 1 (
  shift
  goto require_one
)
exit /b 0
:require_one_missing
echo [error] Need mingw32-make (preferred) or make in PATH.
exit /b 1

:detect_arch
set "ARCH_ID=x86_64"
set "ARCH_GUESS=%PROCESSOR_ARCHITEW6432%"
if "%ARCH_GUESS%"=="" set "ARCH_GUESS=%PROCESSOR_ARCHITECTURE%"
if /I "%ARCH_GUESS%"=="x86" set "ARCH_ID=x86_32"
if /I "%ARCH_GUESS%"=="i386" set "ARCH_ID=x86_32"
if /I "%ARCH_GUESS%"=="ARM64" set "ARCH_ID=arm64"
if /I "%ARCH_GUESS%"=="ARM" set "ARCH_ID=arm32"
exit /b 0

:ensure_dir
if exist "%~1" goto ensure_dir_done
mkdir "%~1" >nul
:ensure_dir_done
exit /b 0

:cmake_configure
set "SRC=%~1"
set "BDIR=%~2"
set "EXTRA=%~3"
call :ensure_dir "%BDIR%"
cmake -S "%SRC%" -B "%BDIR%" -G "%GENERATOR%" -D BUILD_SHARED_LIBS=OFF -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_BUILD_TYPE=Release %EXTRA%
exit /b %ERRORLEVEL%

:cmake_build_targets
set "BDIR=%~1"
set "TARGETS=%~2"
for %%T in (%TARGETS%) do (
  cmake --build "%BDIR%" --config Release --target %%T --parallel %JOBS%
  if errorlevel 1 exit /b 1
)
exit /b 0

:copy_single
set "BDIR=%~1"
set "EXE=%~2"
set "DEST=%~3"
set "C1=%BDIR%\%EXE%"
set "C2=%BDIR%\bin\%EXE%"
set "C3=%BDIR%\Release\%EXE%"
set "C4=%BDIR%\bin\Release\%EXE%"
if exist "%C1%" (
  copy /Y "%C1%" "%DEST%\%EXE%" >nul
  echo Copied %EXE% -> %DEST%
  exit /b 0
)
if exist "%C2%" (
  copy /Y "%C2%" "%DEST%\%EXE%" >nul
  echo Copied %EXE% -> %DEST%
  exit /b 0
)
if exist "%C3%" (
  copy /Y "%C3%" "%DEST%\%EXE%" >nul
  echo Copied %EXE% -> %DEST%
  exit /b 0
)
if exist "%C4%" (
  copy /Y "%C4%" "%DEST%\%EXE%" >nul
  echo Copied %EXE% -> %DEST%
  exit /b 0
)
echo [error] Unable to locate %EXE% under %BDIR%.
exit /b 1

:copy_runtime
set "TARGET_DIR=%~1"
for %%D in (%RUNTIME_DLLS%) do (
  call :find_runtime "%%~D" DLL_PATH
  if "!DLL_PATH!"=="" (
    echo [warn] Unable to locate %%~D in PATH.
  ) else (
    copy /Y "!DLL_PATH!" "%TARGET_DIR%\%%~D" >nul
  )
)
exit /b 0

:find_runtime
set "DLL_NAME=%~1"
set "DLL_RESULT="
for /f "delims=" %%P in ('where %DLL_NAME% 2^>nul') do (
  if "!DLL_RESULT!"=="" set "DLL_RESULT=%%~fP"
)
set "%~2=%DLL_RESULT%"
exit /b 0

:build_llama
set "SRC=%ROOT%\llama.cpp"
:llama_src_check
if exist "%SRC%\CMakeLists.txt" goto llama_src_ok
echo [error] Missing llama.cpp sources at %SRC%.
exit /b 1
:llama_src_ok
set "BDIR=%BUILD_DIR%\llama.cpp"
set "OUTP=%OUT_DIR%\llama.cpp"
set "DEFS=-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_OPENCL=OFF -DSD_VULKAN=OFF -DSD_CUDA=OFF -DSD_OPENCL=OFF -DCMAKE_OBJECT_PATH_MAX=196"
call :cmake_configure "%SRC%" "%BDIR%" "%DEFS%" || exit /b 1
call :cmake_build_targets "%BDIR%" "llama-server llama-quantize" || exit /b 1
call :ensure_dir "%OUTP%"
call :copy_single "%BDIR%" "llama-server.exe" "%OUTP%"
if errorlevel 1 exit /b 1
call :copy_single "%BDIR%" "llama-quantize.exe" "%OUTP%"
if errorlevel 1 exit /b 1
call :copy_runtime "%OUTP%"
exit /b %ERRORLEVEL%

:build_whisper
set "SRC=%ROOT%\whisper.cpp"
:whisper_src_check
if exist "%SRC%\CMakeLists.txt" goto whisper_src_ok
echo [error] Missing whisper.cpp sources at %SRC%.
exit /b 1
:whisper_src_ok
set "BDIR=%BUILD_DIR%\whisper.cpp"
set "OUTP=%OUT_DIR%\whisper.cpp"
set "DEFS=-DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_OPENCL=OFF -DSD_VULKAN=OFF -DSD_CUDA=OFF -DSD_OPENCL=OFF"
call :cmake_configure "%SRC%" "%BDIR%" "%DEFS%" || exit /b 1
call :cmake_build_targets "%BDIR%" "whisper-cli" || exit /b 1
call :ensure_dir "%OUTP%"
call :copy_single "%BDIR%" "whisper-cli.exe" "%OUTP%"
if errorlevel 1 exit /b 1
call :copy_runtime "%OUTP%"
exit /b %ERRORLEVEL%

:build_sd
set "SRC=%ROOT%\stable-diffusion.cpp"
:sd_src_check
if exist "%SRC%\CMakeLists.txt" goto sd_src_ok
echo [error] Missing stable-diffusion.cpp sources at %SRC%.
exit /b 1
:sd_src_ok
set "BDIR=%BUILD_DIR%\stable-diffusion.cpp"
set "OUTP=%OUT_DIR%\stable-diffusion.cpp"
set "DEFS=-DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_OPENCL=OFF -DSD_VULKAN=OFF -DSD_CUDA=OFF -DSD_OPENCL=OFF"
call :cmake_configure "%SRC%" "%BDIR%" "%DEFS%" || exit /b 1
call :cmake_build_targets "%BDIR%" "sd" || exit /b 1
call :ensure_dir "%OUTP%"
call :copy_single "%BDIR%" "sd.exe" "%OUTP%"
if errorlevel 1 exit /b 1
call :copy_runtime "%OUTP%"
exit /b %ERRORLEVEL%

:build_tts
set "SRC=%ROOT%\tts.cpp"
:tts_src_check
if exist "%SRC%\CMakeLists.txt" goto tts_src_ok
echo [error] Missing tts.cpp sources at %SRC%.
exit /b 1
:tts_src_ok
set "BDIR=%BUILD_DIR%\tts.cpp"
set "OUTP=%OUT_DIR%\tts.cpp"
set "DEFS=-DTTS_BUILD_EXAMPLES=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_OPENCL=OFF"
call :cmake_configure "%SRC%" "%BDIR%" "%DEFS%" || exit /b 1
call :cmake_build_targets "%BDIR%" "tts-cli" || exit /b 1
call :ensure_dir "%OUTP%"
call :copy_single "%BDIR%" "tts-cli.exe" "%OUTP%"
if errorlevel 1 exit /b 1
call :copy_runtime "%OUTP%"
exit /b %ERRORLEVEL%

