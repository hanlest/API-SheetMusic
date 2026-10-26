@echo off
rem Arranca MuScriptor API con PM2 (prepara GPU/.venv con start-api.bat, luego muscriptor.exe).
rem Requiere Node.js y: npm install -g pm2

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
goto :main

:load_dotenv
if not exist ".env" exit /b 0
for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
    if /i "%%a"=="HF_TOKEN" set "HF_TOKEN=%%b"
)
if defined HF_TOKEN (
    if "!HF_TOKEN:~0,1!"=="""" set "HF_TOKEN=!HF_TOKEN:~1,-1!"
)
exit /b 0

:main
set "ARG1=%~1"
if /I "%ARG1%"=="__run__" set "ARG1="

set "APP_NAME=muscriptor-api"
set "UV_LINK_MODE=copy"

call :load_dotenv

where pm2 >nul 2>&1
if errorlevel 1 (
    echo No se encontro "pm2" en el PATH.
    echo Instalalo con: npm install -g pm2
    if /I not "%ARG1%"=="nopause" pause
    exit /b 1
)

if not exist "ecosystem.config.cjs" (
    echo Falta ecosystem.config.cjs en %CD%
    if /I not "%ARG1%"=="nopause" pause
    exit /b 1
)

rem Libera torch en disco antes de uv pip (evita "Acceso denegado" si la API estaba en PM2).
call pm2 describe "%APP_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo Deteniendo "%APP_NAME%" en PM2 antes de preparar el entorno...
    call pm2 stop "%APP_NAME%" >nul 2>&1
    call pm2 delete "%APP_NAME%" >nul 2>&1
)

echo Preparando entorno ^(GPU, torch, .venv^)...
rem Tesla P40 / Pascal: no probar cu128 en cada arranque (~60s de reinstalacion inutil).
set "USE_GPU=legacy"
set "PREPARE_ONLY=1"
set "PM2_NO_PAUSE=1"
call "%~dp0start-api.bat"
set "PREPARE_ERR=!ERRORLEVEL!"
set "PREPARE_ONLY="
if not "!PREPARE_ERR!"=="0" goto :fail

if not exist ".venv\Scripts\muscriptor.exe" (
    echo No se encontro muscriptor.exe. Ejecuta start-api.bat una vez manualmente.
    goto :fail
)

echo Iniciando "%APP_NAME%" con ecosystem.config.cjs...
call pm2 start ecosystem.config.cjs --update-env
if errorlevel 1 goto :fail

echo.
call pm2 status "%APP_NAME%"
echo.
echo   Logs:    pm2 logs %APP_NAME%
echo   Detener: pm2-stop.bat
echo.
if /I not "%ARG1%"=="nopause" pause
endlocal
exit /b 0

:fail
echo.
echo Error al iniciar con PM2.
if /I not "%ARG1%"=="nopause" pause
exit /b 1
