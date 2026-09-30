@echo off
rem Only needed to regenerate assets\avatar\idle.mp4. Then run: python scripts\windows\wan_idle.py
call "%~dp0..\..\config\paths.bat"
cd /d "%COMFY_DIR%"
"%COMFY_DIR%\python_embeded\python.exe" -s ComfyUI\main.py --windows-standalone-build --port %COMFY_PORT%
