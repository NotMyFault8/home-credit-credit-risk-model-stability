FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "home_credit.api:app", "--host", "0.0.0.0", "--port", "8000"]
