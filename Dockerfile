FROM python:3.12-slim

WORKDIR /app

# Install build dependencies required by faiss-cpu / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit listens on 8501 by default
EXPOSE 8501

# Create the faiss_store directory on first run if missing
RUN mkdir -p faiss_store

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
