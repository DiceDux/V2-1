@echo off
chcp 65001 > nul
cd /d "%~dp0"
call .venv\Scripts\activate
echo ✅ محیط فعال شد...
echo 🧠 در حال آموزش مدل...
python train_ai_model_multi_advanced.py
pause

