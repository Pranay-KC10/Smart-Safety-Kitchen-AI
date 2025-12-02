
# Build command: docker build -t kitchen-ai-stage3 .
# Run command: docker run kitchen-ai-stage3
# ==================================================================

# STEP 1: Choose a base image
# This is like choosing which OS + software to start with
# python:3.10-slim = Linux with Python 3.10 already installed
FROM python:3.10-slim

# STEP 2: Set working directory inside the container
# Think of this as "cd /app" - all commands run from here
WORKDIR /app

# STEP 3: Copy your requirements file
# Copy from your laptop → into the container
COPY requirements.txt .

# STEP 4: Install Python dependencies
# Run pip install inside the container
RUN pip install --no-cache-dir -r requirements.txt

# STEP 5: Copy all your code files
# Copy everything from src/ → /app in container
COPY . .

# STEP 6: Define what command runs when container starts
# When you do "docker run", this is what executes
CMD ["python", "main.py"]

