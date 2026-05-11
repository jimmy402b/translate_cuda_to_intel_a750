@echo off
echo ========================================
echo Phase 3: Fused Hash Encoding - Build & Test
echo ========================================
echo.

REM Source Intel oneAPI environment
echo [1/4] Setting up oneAPI environment...
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to source oneAPI environment
    exit /b 1
)
echo oneAPI environment loaded.

REM Activate Python venv
echo [2/4] Activating Python venv...
call Z:\nerf_and_3dGS\venv_xpu\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate venv
    exit /b 1
)

REM Show icx version
echo.
echo DPC++ compiler:
icx --version 2>&1 | findstr /C:"DPC++" /C:"version"

REM Check PyTorch
echo.
echo PyTorch version:
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'XPU available: {torch.xpu.is_available()}')"

REM Run correctness test
echo.
echo [3/4] Running correctness test...
cd /d Z:\nerf_and_3dGS\HashNeRF-pytorch
python tests/test_hash_encode.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo TESTS FAILED!
    exit /b 1
)

echo.
echo [4/4] All tests passed!
echo ========================================
