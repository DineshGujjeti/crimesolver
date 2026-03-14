FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV and Torch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (improves Docker cache)
COPY backend/requirements.txt ./backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source code
COPY backend/ ./backend/

# Copy AI models
COPY models/ ./models/

# Move into backend folder
WORKDIR /app/backend

# Create runtime folders
RUN mkdir -p uploads reports

# Render provides PORT env variable
ENV PORT=8000

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]