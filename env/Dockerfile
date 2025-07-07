FROM python:3.10-slim

# set workdir
WORKDIR /ball_vision

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinerama0 \
    x11-apps \
    tk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ${WORKDIR}/requirements.txt
COPY ../models/* /opt/balls/*
COPY ../nuclio/function.yaml /opt/nuclio/function.yaml

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${WORKDIR}/requirements.txt

ENV DISPLAY=:${DISPLAY} 

CMD ["python", "scripts/run_img.py" , "--image", "ball_small_1.jpg", "--model", "ball_weights_cheated.pt"]