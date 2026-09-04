FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY README.md ./

RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "llm_evals.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
