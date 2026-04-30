@echo off
REM Activate MSVC + CUDA so that PyTorch's JIT extension build works on Windows.
REM Use:  cmd /c "scripts\win_env.bat && python -c '...'"
REM       cmd /c "scripts\win_env.bat && ns-train splatfacto ..."
REM
REM This script is idempotent: vcvars64 sets a flag and is a no-op on re-entry.

if not defined VSINSTALLDIR (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL
)

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=%CUDA_HOME%"
set "CUDA_PATH_V12_9=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"

REM RTX 5070 Laptop = sm_120 (Blackwell). Add older arches for fallback wheels.
if "%TORCH_CUDA_ARCH_LIST%"=="" set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
set "FORCE_CUDA=1"

REM Force UTF-8 for nerfstudio/rich (otherwise emoji like 🎉 in the
REM "Training Finished" panel raise UnicodeEncodeError on Windows GBK codepage
REM and tear the subprocess down with exit 1 even after a successful train run).
set "PYTHONIOENCODING=utf-8"
chcp 65001 >NUL
