@echo off
setlocal
call .venv\Scripts\activate.bat
python scripts\manage.py ui --host 0.0.0.0 --port 8090 --output-root outputs\demo_ui
endlocal
