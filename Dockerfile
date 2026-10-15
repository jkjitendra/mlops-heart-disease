FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-serving.txt .

RUN python -m pip install --no-cache-dir -r requirements-serving.txt

COPY src/api.py ./src/api.py

RUN useradd --create-home --uid 10001 appuser

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
