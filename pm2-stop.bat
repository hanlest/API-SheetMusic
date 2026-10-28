@echo off
rem Detiene el proceso PM2 API-SheetMusic (creado con pm2-start.bat).

setlocal EnableExtensions
cd /d "%~dp0"

set "ARG1=%~1"
if /I "%ARG1%"=="__run__" set "ARG1="

set "APP_NAME=API-SheetMusic"

where pm2 >nul 2>&1
if errorlevel 1 (
    echo No se encontro "pm2" en el PATH.
    if /I not "%ARG1%"=="nopause" pause
    exit /b 1
)

call pm2 describe "%APP_NAME%" >nul 2>&1
if errorlevel 1 (
    echo No hay ningun proceso PM2 llamado "%APP_NAME%".
    if /I not "%ARG1%"=="nopause" pause
    exit /b 0
)

echo Deteniendo "%APP_NAME%"...
call pm2 stop "%APP_NAME%"
if errorlevel 1 (
    echo No se pudo detener el proceso.
    if /I not "%ARG1%"=="nopause" pause
    exit /b 1
)

echo.
call pm2 status "%APP_NAME%"
echo.
echo Para volver a arrancar: pm2-start.bat
echo Para quitarlo de PM2:     pm2 delete %APP_NAME%
echo.
if /I not "%ARG1%"=="nopause" pause
endlocal
