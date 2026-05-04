FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libpq-dev gcc curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8002
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
