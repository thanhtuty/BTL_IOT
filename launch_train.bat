@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay .venv. Hay cai Python va tao moi truong ao truoc.
  echo py -3.11 -m venv .venv
  echo .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo .venv\Scripts\python.exe -m pip install streamlit
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run train_app.py --server.address 127.0.0.1
pause
