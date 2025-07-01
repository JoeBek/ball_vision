# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /ball_vision

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinerama0 \
    x11-apps \
    tk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ${WORKDIR}/requirements.txt


# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${WORKDIR}/requirements.txt

ENV DISPLAY=:${DISPLAY} 

# Set the default command to run your script
CMD ["python", "scripts/run_img.py" , "--image", "ball_small_1.jpg", "--model", "best.pt"]