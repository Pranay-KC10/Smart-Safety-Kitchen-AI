# Classifier Container (Stage 2)
# Classifies stove (ON/OFF), knife (in-use/unattended), pan status

FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir ultralytics opencv-python numpy pillow

# Copy classifier code
COPY run_classifier.py .

# Create output directories
RUN mkdir -p /outputs /shared

# Run classifier
CMD ["python", "run_classifier.py"]
