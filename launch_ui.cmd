@echo off
setlocal
call .venv\Scripts\activate.bat
python scripts\manage.py ui --host 127.0.0.1 --port 8090 --output-root outputs\demo_ui
endlocal
