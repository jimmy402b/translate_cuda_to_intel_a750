@echo off
echo ========================================
echo Phase 3: Fused Hash Encoding - Build ^& Test
echo ========================================
echo.

REM --- 1. MSVC: add bin to PATH, then source vcvarsall for INCLUDE/LIB ---
echo [1/4] Setting up MSVC environment...
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to source MSVC environment
    exit /b 1
)
echo MSVC OK

REM --- 2. oneAPI: manual paths (skip setvars.bat to avoid component errors) ---
echo [2/4] Setting up oneAPI + SYCL runtime paths...
REM Compiler bin: icx.exe + sycl9.dll (compiler 2026's own SYCL runtime)
set "PATH=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin;%PATH%"
set "PATH=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib;%PATH%"
REM intel-sycl-rt: sycl8.dll (PyTorch XPU's SYCL runtime, for c10_xpu.dll etc.)
set "PATH=Z:\nerf_and_3dGS\venv_xpu\Library\bin;%PATH%"
REM SYCL_HOME must point to the COMPILER (not intel-sycl-rt) --
REM both SYCL headers and sycl.lib must match icx version (2026).
REM The PYD will link against sycl9.dll (from compiler/bin, already in PATH).
set "SYCL_HOME=C:\Program Files (x86)\Intel\oneAPI\compiler\latest"
set "TORCH_XPU_ARCH_LIST="
set "INCLUDE=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207\include;%INCLUDE%"
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;%INCLUDE%"
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;%INCLUDE%"
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;%INCLUDE%"
set "INCLUDE=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include;%INCLUDE%"
set "LIB=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64;%LIB%"
set "LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;%LIB%"
set "LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;%LIB%"
set "LIB=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib;%LIB%"
echo oneAPI + SYCL runtime OK

REM --- 3. Activate venv ---
echo [3/4] Activating Python venv...
call Z:\nerf_and_3dGS\venv_xpu\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate venv
    exit /b 1
)

REM --- Show versions ---
echo.
echo Compilers:
where cl 2>&1
where icx 2>&1
echo SYCL_HOME=%SYCL_HOME%
echo.
echo PyTorch:
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'XPU available: {torch.xpu.is_available()}')"
echo.

REM --- 4. Run test ---
echo [4/4] Running import check + correctness test...
cd /d Z:\nerf_and_3dGS\HashNeRF-pytorch
python sycl_ops\check_import.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo TESTS FAILED - see output above for details
    echo ========================================
    exit /b 1
)

echo.
echo ========================================
echo [OK] All tests passed!
echo ========================================
exit /b 0
