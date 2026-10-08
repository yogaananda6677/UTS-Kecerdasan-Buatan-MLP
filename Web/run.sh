#!/bin/bash
cd "$(dirname "$0")"
echo "=================================================="
echo " Starting CardioMLP Analytics Server (FastAPI)    "
echo " URL: http://localhost:5000                       "
echo " API Docs: http://localhost:5000/docs             "
echo "=================================================="
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
