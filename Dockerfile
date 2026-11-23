
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app/ ./app/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
