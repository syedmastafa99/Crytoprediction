# requirements.txt
flask==2.3.3
pandas==2.1.0
numpy==1.25.2
requests==2.31.0
scikit-learn==1.3.0
plotly==5.17.0
gunicorn==21.2.0

# render.yaml
services:
  - type: web
    name: crypto-predictor
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0

# .gitignore
__pycache__/
*.py[cod]
*$py.class
venv/
.env
.DS_Store
