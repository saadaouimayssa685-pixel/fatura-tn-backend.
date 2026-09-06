# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/appuser
ENV PADDLEX_HOME=/home/appuser/.paddlex
ENV PADDLEOCR_HOME=/home/appuser/.paddleocr
ENV UPLOAD_FOLDER=/app/data/uploads
ENV INVOICE_DB_PATH=/app/data/invoice_storage/invoices.sqlite3
ENV INVOICE_STORAGE_FOLDER=/app/data/invoice_storage/documents
ENV RAG_DB_PATH=/app/data/rag_storage/rag.sqlite3
ENV RAG_STORAGE_FOLDER=/app/data/rag_storage/documents
ENV YOLO_MODEL_PATH=/app/models/extraction/best.pt
ENV TESSDATA_DIR=/app/backend/tessdata

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Tesseract OCR and language packs
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # Other utilities
    wget \
    curl \
    git \
    build-essential \
    pkg-config \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /app/backend

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/models/extraction /app/assets/fonts

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for YOLO and machine learning
RUN pip install --no-cache-dir ultralytics

# Copy application code
COPY backend/ .
COPY models/ /app/models/
COPY assets/ /app/assets/

# Create a non-root user for security and set up directories
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser

# Create PaddleOCR directories with proper permissions
RUN mkdir -p /home/appuser/.paddlex/temp \
    && mkdir -p /home/appuser/.paddleocr \
    && mkdir -p /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
