@echo off
REM start.bat - Windows launcher. Runs start.ps1 with the execution policy
REM bypassed so you don't have to change PowerShell settings.
REM Double-click this file, or run "start.bat" from a terminal in this folder.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
pause
