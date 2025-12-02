# ==================== DOCKERFILE FOR STAGE 3 ====================
# This file tells Docker how to create a "container" for your code
#
# Think of it like a recipe:
# 1. Start with a base (Python installed)
# 2. Add your code files
# 3. Install dependencies
# 4. Tell it what to run
#
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

# ==================================================================
# DOCKER BASICS EXPLAINED:
#
# What happens when you run this?
# 1. Docker reads this file top to bottom
# 2. Creates a new "virtual computer" (container)
# 3. Installs Python 3.10 in it
# 4. Copies your code into it
# 5. Installs requirements
# 6. Runs main.py
#
# Why is this useful?
# - Your friend can run your code without installing anything!
# - Just needs Docker installed, then: docker run kitchen-ai-stage3
# - Works the same on Windows, Mac, Linux, Jetson Nano
#
# Docker vs Regular Python:
# Regular: "python main.py" (needs Python installed on your laptop)
# Docker: "docker run image-name" (Python is INSIDE the container)
# ==================================================================
