FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    pandas

# Copy detector script
COPY scripts/run-detector.sh .

# Make script executable
RUN chmod +x run-detector.sh

# Set entrypoint
ENTRYPOINT ["/bin/bash"]