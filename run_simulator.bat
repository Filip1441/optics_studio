@echo off
title Optics Studio
echo Launching Optics Studio...
echo Monitoring Optical Engine...

if not exist venv (
    echo [ERROR] Srodowisko wirtualne nie istnieje!
    echo Uruchom najpierw setup.bat
    pause
    exit /b
)

call venv\Scripts\activate
python gui_app.py
pause
