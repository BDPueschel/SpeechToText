$WshShell = New-Object -ComObject WScript.Shell
$StartupPath = [System.IO.Path]::Combine($env:APPDATA, "Microsoft\Windows\Start Menu\Programs\Startup\WhisperType.lnk")
$Shortcut = $WshShell.CreateShortcut($StartupPath)
$Shortcut.TargetPath = "C:\Users\Brian\Code Projects\SpeechToText\launch_whisper.bat"
$Shortcut.WorkingDirectory = "C:\Users\Brian\Code Projects\SpeechToText"
$Shortcut.WindowStyle = 7  # Minimized
$Shortcut.Save()
Write-Host "Startup shortcut created at: $StartupPath"
