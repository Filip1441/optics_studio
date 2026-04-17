@echo off
echo ==========================================
echo   Optics Studio - Environment Setup
echo ==========================================

echo.
echo [1/3] Tworzenie srodowiska wirtualnego (venv)...
python -m venv venv

echo.
echo [2/3] Instalacja zaleznosci z requirements.txt...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [3/3] Konfiguracja zakonczona!
echo Mozesz teraz uruchomic program za pomoca run_simulator.bat
echo.
pause
