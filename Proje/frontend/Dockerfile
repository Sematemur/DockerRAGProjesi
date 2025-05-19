# frontend/Dockerfile
FROM python:3.10

WORKDIR /gui

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "gui.py", "--server.port=8501", "--server.address=0.0.0.0"]
