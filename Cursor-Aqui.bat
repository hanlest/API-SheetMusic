@echo off
rem Abre Cursor en la carpeta donde esta este .bat (perfil Personal).
rem Copia este archivo a cualquier proyecto bajo D:\PROYECTOS\Personal.

setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "CURSOR=%LOCALAPPDATA%\Programs\cursor\Cursor.exe"
if not exist "%CURSOR%" (
    echo No se encontro Cursor en: %CURSOR%
    pause
    exit /b 1
)

start "" "%CURSOR%" ^
  --user-data-dir="C:\CursorProfiles\Personal" ^
  --extensions-dir="C:\CursorProfiles\Personal\extensions" ^
  --classic ^
  "%ROOT%"
