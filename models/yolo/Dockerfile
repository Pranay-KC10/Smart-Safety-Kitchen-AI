# YOLO Object Detection Container (Stage 1)
# Detects kitchen objects: person, stove, knife, pan, fire, smoke

FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir ultralytics opencv-python numpy

# Copy detector code
COPY run_detector.py .

# Create output directories
RUN mkdir -p /outputs/crops /shared

# Run detector
CMD ["python", "run_detector.py"]
