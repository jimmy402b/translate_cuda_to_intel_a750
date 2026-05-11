@echo off
echo Setting up environment...

REM Set up Visual Studio Build Tools paths (vcvarsall.bat misses MSVC toolchain when vswhere.exe not on PATH)
set "VSINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
set "MSVC_VER=14.44.35207"
set "WKITS_VER=10.0.22621.0"

REM Windows Kits
set "PATH=%VSINSTALLDIR%\MSBuild\Current\Bin\amd64;%VSINSTALLDIR%\Common7\IDE;%VSINSTALLDIR%\Common7\Tools;%PATH%"

REM MSVC toolchain
set "MSVC_BIN=%VSINSTALLDIR%\VC\Tools\MSVC\%MSVC_VER%\bin\HostX64\x64"
set "PATH=%MSVC_BIN%;%PATH%"
set "INCLUDE=%VSINSTALLDIR%\VC\Tools\MSVC\%MSVC_VER%\include;%VSINSTALLDIR%\VC\Tools\MSVC\%MSVC_VER%\atlmfc\include;%INCLUDE%"
set "LIB=%VSINSTALLDIR%\VC\Tools\MSVC\%MSVC_VER%\lib\x64;%VSINSTALLDIR%\VC\Tools\MSVC\%MSVC_VER%\atlmfc\lib\x64;%LIB%"

REM Windows Kits
set "PATH=C:\Program Files (x86)\Windows Kits\10\bin\%WKITS_VER%\x64;%PATH%"
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\include\%WKITS_VER%\ucrt;C:\Program Files (x86)\Windows Kits\10\include\%WKITS_VER%\um;C:\Program Files (x86)\Windows Kits\10\include\%WKITS_VER%\shared;C:\Program Files (x86)\Windows Kits\10\include\%WKITS_VER%\winrt;C:\Program Files (x86)\Windows Kits\10\include\%WKITS_VER%\cppwinrt;%INCLUDE%"
set "LIB=C:\Program Files (x86)\Windows Kits\10\lib\%WKITS_VER%\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\lib\%WKITS_VER%\um\x64;%LIB%"

REM Intel oneAPI DPC++ compiler
set "PATH=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin;%PATH%"
set "LIB=C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0\lib;%LIB%"

echo.
echo Checking link.exe...
where link
echo.
echo Checking icx...
where icx
echo.

cd /d Z:\nerf_and_3dGS\HashNeRF-pytorch\sycl_ops
icx -fsycl test_dpcpp.cpp -o test_dpcpp.exe
if %ERRORLEVEL% NEQ 0 (
    echo COMPILATION FAILED with exit code %ERRORLEVEL%
    exit /b 1
)
echo COMPILATION SUCCESS
echo.
echo Running test_dpcpp.exe...
echo.
.\test_dpcpp.exe
echo.
echo EXIT CODE: %ERRORLEVEL%
