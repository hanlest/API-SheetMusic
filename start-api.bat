@echo off
rem Arranca el servidor HTTP de MuScriptor (API + interfaz web si esta compilada).
rem Requiere uv: https://docs.astral.sh/uv/getting-started/installation/
rem
rem USE_GPU:
rem   0      = solo CPU
rem   1      = CUDA reciente cu128, RTX 20xx+ / sm_75+
rem   legacy = CUDA antigua cu118, Tesla P40 / Pascal
rem   auto   = prueba cu128, luego cu118, luego CPU

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Evita el aviso de hardlink cuando cache y .venv estan en discos distintos.
set "UV_LINK_MODE=copy"

set "HOST=127.0.0.1"
set "PORT=8222"
if not defined USE_GPU set "USE_GPU=auto"
set "SERVE_DEVICE=cpu"
set "GPU_LABEL=CPU"

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "VENV_MUSCRIPTOR=%CD%\.venv\Scripts\muscriptor.exe"

call :load_dotenv

where uv >nul 2>&1
if errorlevel 1 (
    echo No se encontro "uv" en el PATH.
    echo Instalalo y vuelve a ejecutar este archivo.
    call :maybe_pause
    exit /b 1
)

if not exist "%VENV_PY%" (
    echo Creando entorno virtual - uv sync...
    uv sync
    if errorlevel 1 (
        echo uv sync fallo.
        call :maybe_pause
        exit /b 1
    )
)

if /I "!USE_GPU!"=="0" goto :print_banner

set "UV_NO_SYNC=1"

if /I "!USE_GPU!"=="legacy" goto :setup_legacy
if /I "!USE_GPU!"=="1" goto :setup_cu128

rem USE_GPU=auto
call :ensure_cuda_torch cu128
if errorlevel 1 goto :try_legacy_or_cpu
call :probe_cuda_runtime
if not errorlevel 1 (
    set "SERVE_DEVICE=cuda"
    set "GPU_LABEL=CUDA PyTorch cu128"
    goto :print_banner
)

echo.
echo La GPU no es compatible con PyTorch cu128 - se necesita sm_75 o superior.
echo Probando PyTorch cu118 para Pascal/Maxwell, por ejemplo Tesla P40...
call :ensure_cuda_torch cu118
if errorlevel 1 goto :fallback_cpu
call :probe_cuda_runtime
if not errorlevel 1 (
    set "SERVE_DEVICE=cuda"
    set "GPU_LABEL=CUDA legacy PyTorch cu118"
    goto :print_banner
)
goto :fallback_cpu

:setup_cu128
call :ensure_cuda_torch cu128
if errorlevel 1 goto :gpu_failed
call :probe_cuda_runtime
if errorlevel 1 goto :gpu_failed
set "SERVE_DEVICE=cuda"
set "GPU_LABEL=CUDA PyTorch cu128"
goto :print_banner

:setup_legacy
call :ensure_cuda_torch cu118
if errorlevel 1 goto :gpu_failed
call :probe_cuda_runtime
if errorlevel 1 goto :gpu_failed
set "SERVE_DEVICE=cuda"
set "GPU_LABEL=CUDA legacy PyTorch cu118"
goto :print_banner

:try_legacy_or_cpu
call :ensure_cuda_torch cu118
if errorlevel 1 goto :fallback_cpu
call :probe_cuda_runtime
if not errorlevel 1 (
    set "SERVE_DEVICE=cuda"
    set "GPU_LABEL=CUDA legacy PyTorch cu118"
    goto :print_banner
)
goto :fallback_cpu

:fallback_cpu
echo.
echo [Aviso] No se pudo usar la GPU; el modelo correra en CPU - mas lento.
echo         Para forzar CPU: set USE_GPU=0 en este .bat
echo.
set "SERVE_DEVICE=cpu"
set "GPU_LABEL=CPU fallback"
goto :print_banner

:gpu_failed
call :maybe_pause
exit /b 1

:maybe_pause
if /I "%PM2_NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0

:print_banner
echo.
echo MuScriptor API
echo   URL:     http://!HOST!:!PORT!/
echo   Swagger: http://!HOST!:!PORT!/docs
echo   Compute: !GPU_LABEL! - device !SERVE_DEVICE!
echo.

call :warn_huggingface

if /I "!PREPARE_ONLY!"=="1" goto :pm2_prepare_done

if not exist "%VENV_MUSCRIPTOR%" (
    echo No se encontro muscriptor.exe en .venv. Ejecuta: uv sync
    call :maybe_pause
    exit /b 1
)

"%VENV_MUSCRIPTOR%" serve --host %HOST% --port %PORT% --device %SERVE_DEVICE%

if errorlevel 1 (
    echo.
    echo El servidor termino con error.
    call :maybe_pause
    exit /b 1
)

endlocal
exit /b 0

:pm2_prepare_done
echo !SERVE_DEVICE!>"%CD%\.pm2-device"
echo Preparacion PM2 OK - device !SERVE_DEVICE!
exit /b 0

:load_dotenv
if not exist ".env" exit /b 0
for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
    if /i "%%a"=="HF_TOKEN" set "HF_TOKEN=%%b"
)
if defined HF_TOKEN (
    if "!HF_TOKEN:~0,1!"=="""" set "HF_TOKEN=!HF_TOKEN:~1,-1!"
)
exit /b 0

:warn_huggingface
if defined HF_TOKEN exit /b 0
if exist "%USERPROFILE%\.cache\huggingface\token" exit /b 0
echo [Aviso] Falta autenticacion en Hugging Face para descargar el modelo:
echo   1. Acepta la licencia en https://huggingface.co/MuScriptor/muscriptor-medium
echo   2. Crea un token de lectura y ponlo en .env como HF_TOKEN=hf_...
echo.
exit /b 0

:ensure_cuda_torch
set "TORCH_BACKEND=%~1"
if "!TORCH_BACKEND!"=="" set "TORCH_BACKEND=cu128"

set "TORCH_OK=0"
set "AUDIO_OK=0"
call :get_torch_version
echo !TORCH_VER! | findstr /C:"+!TORCH_BACKEND!" >nul 2>&1
if not errorlevel 1 set "TORCH_OK=1"
call :get_torchaudio_version
echo !TORCHAUDIO_VER! | findstr /C:"+!TORCH_BACKEND!" >nul 2>&1
if not errorlevel 1 set "AUDIO_OK=1"
if "!TORCH_OK!"=="1" if "!AUDIO_OK!"=="1" exit /b 0

echo.
echo Instalando PyTorch + torchaudio con CUDA !TORCH_BACKEND! en .venv - puede tardar, ~2 GB...
set "UV_TORCH_BACKEND=!TORCH_BACKEND!"
if /I "!TORCH_BACKEND!"=="cu118" (
    uv pip install --python .venv "torch>=2.3.0,<2.8.0" "torchaudio>=2.3.0,<2.8.0" --torch-backend=cu118 --reinstall-package torch --reinstall-package torchaudio
) else (
    uv pip install --python .venv torch torchaudio --torch-backend=!TORCH_BACKEND! --reinstall-package torch --reinstall-package torchaudio
)
if errorlevel 1 exit /b 1

call :get_torch_version
call :get_torchaudio_version
echo !TORCH_VER! | findstr /C:"+!TORCH_BACKEND!" >nul 2>&1
if errorlevel 1 (
    echo Error: se esperaba torch+!TORCH_BACKEND! pero hay !TORCH_VER!.
    exit /b 1
)
echo !TORCHAUDIO_VER! | findstr /C:"+!TORCH_BACKEND!" >nul 2>&1
if errorlevel 1 (
    echo Error: se esperaba torchaudio+!TORCH_BACKEND! pero hay !TORCHAUDIO_VER!.
    exit /b 1
)
echo PyTorch listo: !TORCH_VER! / torchaudio !TORCHAUDIO_VER!
exit /b 0

:probe_cuda_runtime
set "CUDA_PROBE_FILE=%TEMP%\muscriptor-cuda-probe-%RANDOM%.txt"
"%VENV_PY%" "%CD%\scripts\cuda_probe.py" >"%CUDA_PROBE_FILE%" 2>nul
if not exist "%CUDA_PROBE_FILE%" exit /b 1
findstr /B /C:"OK;" "%CUDA_PROBE_FILE%" >nul 2>&1
set /p CUDA_PROBE=<"%CUDA_PROBE_FILE%"
if errorlevel 1 (
    echo Prueba CUDA: !CUDA_PROBE!
    del "%CUDA_PROBE_FILE%" >nul 2>&1
    exit /b 1
)
echo Prueba CUDA: !CUDA_PROBE!
del "%CUDA_PROBE_FILE%" >nul 2>&1
exit /b 0

:get_torch_version
set "TORCH_VER="
set "TORCH_VER_FILE=%TEMP%\muscriptor-torch-ver-%RANDOM%.txt"
"%VENV_PY%" -c "import torch; print(torch.__version__)" >"%TORCH_VER_FILE%" 2>nul
if exist "%TORCH_VER_FILE%" (
    set /p TORCH_VER=<"%TORCH_VER_FILE%"
    del "%TORCH_VER_FILE%" >nul 2>&1
)
exit /b 0

:get_torchaudio_version
set "TORCHAUDIO_VER="
set "TORCHAUDIO_VER_FILE=%TEMP%\muscriptor-torchaudio-ver-%RANDOM%.txt"
"%VENV_PY%" -c "import torchaudio; print(torchaudio.__version__)" >"%TORCHAUDIO_VER_FILE%" 2>nul
if exist "%TORCHAUDIO_VER_FILE%" (
    set /p TORCHAUDIO_VER=<"%TORCHAUDIO_VER_FILE%"
    del "%TORCHAUDIO_VER_FILE%" >nul 2>&1
)
exit /b 0
