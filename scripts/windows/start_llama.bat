@echo off
rem Start llama-server (Qwen3-14B GGUF) on port %LLAMA_PORT%. Keep this window open.
rem Full executable path on purpose: cmd may not search the current directory (NoDefaultCurrentDirectoryInExePath).
call "%~dp0..\..\config\paths.bat"
cd /d "%LLAMA_DIR%"
"%LLAMA_DIR%\llama-server.exe" -m "%LLAMA_MODEL%" %LLAMA_ARGS%
