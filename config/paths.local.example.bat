@echo off
rem Copy to config\paths.local.bat (gitignored) and uncomment what differs on your machine.
rem Called by config\paths.bat after the defaults, so anything here overrides them.

rem set WSL_DISTRO=Ubuntu-24.04
rem set LLAMA_DIR=D:\tools\llama.cpp
rem set LLAMA_MODEL=D:\models\Qwen3-14B-Q4_K_M.gguf
rem set LLAMA_PORT=8090
rem set LLAMA_ARGS=-np 1 -c 8192 -fa on --temp 1.0 --top-p 0.95 --host 0.0.0.0 --port %LLAMA_PORT%
rem set COMFY_DIR=D:\ComfyUI_windows_portable
