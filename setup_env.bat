@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Done. Virtual environment is ready.
pause
