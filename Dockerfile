FROM python:3.12-slim

# Create a non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Install dependencies first (layer-caching friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

USER app

CMD ["python", "-m", "src.main"]
