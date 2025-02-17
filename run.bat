@echo off
echo Instalando dependencias...
pip install -r requirements.txt

echo Iniciando o analisador de imagens...
python main.py
pause