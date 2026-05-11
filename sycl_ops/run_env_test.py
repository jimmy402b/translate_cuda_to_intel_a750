"""Set up the full build environment and run check_import.py."""
import os
import sys
import subprocess

MSVC_BASE = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
MSVC_VER = "14.44.35207"
MSVC_BIN = os.path.join(MSVC_BASE, "VC", "Tools", "MSVC", MSVC_VER, "bin", "HostX64", "x64")
MSVC_INCL = os.path.join(MSVC_BASE, "VC", "Tools", "MSVC", MSVC_VER, "include")
MSVC_LIB = os.path.join(MSVC_BASE, "VC", "Tools", "MSVC", MSVC_VER, "lib", "x64")

WINSDK = r"C:\Program Files (x86)\Windows Kits\10"
WINSDK_VER = "10.0.22621.0"

ONEAPI = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest"
SYCL_RT = r"Z:\nerf_and_3dGS\venv_xpu\Library"

# PATH
os.environ["PATH"] = os.pathsep.join([
    MSVC_BIN,
    os.path.join(ONEAPI, "bin"),
    os.path.join(ONEAPI, "lib"),
    os.path.join(SYCL_RT, "bin"),
    os.environ.get("PATH", ""),
])

# INCLUDE
os.environ["INCLUDE"] = os.pathsep.join([
    MSVC_INCL,
    os.path.join(WINSDK, "Include", WINSDK_VER, "ucrt"),
    os.path.join(WINSDK, "Include", WINSDK_VER, "um"),
    os.path.join(WINSDK, "Include", WINSDK_VER, "shared"),
    os.path.join(ONEAPI, "include"),
    os.environ.get("INCLUDE", ""),
])

# LIB
os.environ["LIB"] = os.pathsep.join([
    MSVC_LIB,
    os.path.join(WINSDK, "Lib", WINSDK_VER, "um", "x64"),
    os.path.join(WINSDK, "Lib", WINSDK_VER, "ucrt", "x64"),
    os.path.join(ONEAPI, "lib"),
    os.environ.get("LIB", ""),
])

os.environ["SYCL_HOME"] = ONEAPI
os.environ["VSCMD_ARG_TGT_ARCH"] = "x64"
os.environ["TORCH_XPU_ARCH_LIST"] = ""

# Run check_import.py
os.chdir(r"Z:\nerf_and_3dGS\HashNeRF-pytorch")
sys.exit(subprocess.call([sys.executable, "sycl_ops/check_import.py"]))
