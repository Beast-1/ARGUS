@echo off
REM Rebuild the standalone ARGUS desktop app into dist_app\ARGUS\
cd /d "%~dp0"
.venv\Scripts\pyinstaller.exe --noconfirm --onedir --windowed --name ARGUS --distpath dist_app --workpath build_app argus_app.py
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo argus.home marker keeps app data in this project folder:
echo %~dp0> dist_app\ARGUS\argus.home
echo Done: dist_app\ARGUS\ARGUS.exe
