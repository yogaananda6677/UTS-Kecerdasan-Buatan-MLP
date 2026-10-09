@echo off
echo ==================================================
echo  Starting CardioMLP Analytics Server (FastAPI)
echo  URL: http://localhost:5000
echo  API Docs: http://localhost:5000/docs
echo ==================================================
cd /d "%~dp0Web"
call ..\.venv\Scripts\activate.bat
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
pause
