@echo off
REM Launches the ARGUS desktop app: FastAPI backend + Tauri/React frontend.
REM The API must run from the repo root — main.py resolves OUTPUT_ROOT relative to CWD.
setlocal

cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%VENV_PY%" (
  echo ARGUS .venv not found. Create it with:
  echo   py -3 -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  goto :done
)

"%VENV_PY%" -c "import fastapi, uvicorn" >nul 2>nul
if errorlevel 1 (
  echo API dependencies missing. Install them with:
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  goto :done
)

REM Shared secret between the backend and the Tauri frontend (service/auth.py /
REM app/src-tauri/src/lib.rs's get_api_token). Generated once here so both the
REM detached "ARGUS API" process and the Tauri dev process below inherit the exact
REM same value; the backend would otherwise generate its own on first import and
REM the frontend would never see it.
for /f %%T in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString(\"N\")"') do set "ARGUS_API_TOKEN=%%T"

echo Starting ARGUS API on http://127.0.0.1:8420 ...
start "ARGUS API" /min "%VENV_PY%" -m uvicorn service.api:app --host 127.0.0.1 --port 8420

echo Starting ARGUS desktop app ...
cd app
call npm run tauri dev

:done
endlocal
