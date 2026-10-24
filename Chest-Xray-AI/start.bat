@echo off
REM One-command start (Windows). First time: start.bat --setup   then: start.bat
cd /d "%~dp0"
where py >nul 2>nul && (py -3 run.py %* & goto :eof)
where python >nul 2>nul && (python run.py %* & goto :eof)
echo Python 3.10-3.12 was not found. Install it from https://www.python.org/downloads/ and tick "Add to PATH".
pause
