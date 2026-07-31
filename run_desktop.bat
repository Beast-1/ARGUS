@echo off
setlocal

cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "ARGUS_REQUIRED_IMPORTS=import requests, dotenv, trimesh, numpy"

set "VIRTUAL_ENV="
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if exist "%VENV_PY%" (
  "%VENV_PY%" -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo ARGUS .venv exists, but its Python is not runnable.
    echo Recreate it with a working Python install:
    echo   py -3 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
  ) else (
    set "VIRTUAL_ENV=%CD%\.venv"
    echo Using ARGUS venv: %CD%\.venv
    "%VENV_PY%" -c "%ARGUS_REQUIRED_IMPORTS%" >nul 2>nul
    if errorlevel 1 (
      echo ARGUS .venv is active, but dependencies are missing.
      echo Install them with:
      echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
      pause
      goto :done
    )
    "%VENV_PY%" desktop_app.py
    goto :done
  )
)

where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "%ARGUS_REQUIRED_IMPORTS%" >nul 2>nul
  if not errorlevel 1 py -3 desktop_app.py
  if not errorlevel 1 goto :done
)

where python >nul 2>nul
if not errorlevel 1 (
  python -c "%ARGUS_REQUIRED_IMPORTS%" >nul 2>nul
  if not errorlevel 1 (
    echo ARGUS .venv not found or broken. Falling back to PATH python.
    python desktop_app.py
  )
  if not errorlevel 1 goto :done
)

echo No runnable ARGUS Python environment found.
echo Recreate this project's venv with a working Python:
echo   C:\Users\darpa\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
pause

:done
endlocal
