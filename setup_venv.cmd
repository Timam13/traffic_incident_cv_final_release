@echo off
setlocal

python -m venv .venv
if errorlevel 1 exit /b 1

call .venv\Scripts\activate.bat
if errorlevel 1 exit /b 1

python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install -e ".[train]"
if errorlevel 1 exit /b 1

echo.
echo Environment is ready.
echo Activate it with:
echo .venv\Scripts\activate.bat

endlocal