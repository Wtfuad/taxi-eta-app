FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app

# libgomp is required by LightGBM; curl is used by the health check
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY taxi_eta ./taxi_eta
COPY app ./app
COPY api ./api
COPY .streamlit ./.streamlit
COPY models ./models

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fs http://localhost:${PORT:-8501}/_stcore/health || curl -fs http://localhost:8000/health || exit 1

# Default runs Streamlit; to run FastAPI instead: docker run -p 8000:8000 taxi-eta uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD streamlit run app/main.py --server.port=${PORT:-8501} --server.address=0.0.0.0
