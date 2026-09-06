FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IP_SAKTI_ENV=production \
    IP_SAKTI_TEST_MODE=false

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data/runtime

EXPOSE 8000
CMD ["uvicorn", "ip_sakti.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
