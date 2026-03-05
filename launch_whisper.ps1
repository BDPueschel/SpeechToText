# launch_whisper.ps1
# Launches whisper_type.py in the background (no Claude Code).
#
# Double-click launch_whisper.bat to run.

# ---------------------------------------------
#  CONFIG
# ---------------------------------------------
$SCRIPT_DIR     = Split-Path -Parent $MyInvocation.MyCommand.Path
$WHISPER_SCRIPT = Join-Path $SCRIPT_DIR "whisper_type.py"
$PYTHON_EXE     = "python"
# ---------------------------------------------

# Kill any existing whisper_type.py instance before starting a new one
Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*whisper_type.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# Launch whisper_type.py as a hidden background process (system tray)
Start-Process $PYTHON_EXE -ArgumentList "`"$WHISPER_SCRIPT`"" -WindowStyle Hidden
