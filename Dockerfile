# Use slim Python image
FROM python:3.10-slim

# Prevent writing pyc files & enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for transformers & sentencepiece
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libsndfile1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest
RUN pip install --upgrade pip

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (NumPy pinned to 1.x for PyTorch compatibility)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend code
COPY app /app/app

# Expose port for Render
EXPOSE 8000

# Run Uvicorn on 0.0.0.0:8000 with 1 worker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
