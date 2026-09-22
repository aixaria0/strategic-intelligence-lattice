FROM python:3.11-slim
WORKDIR /opt/sil
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY app.py ./
RUN pip install --no-cache-dir -e '.[ui]'
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
