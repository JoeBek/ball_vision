#!/bin/bash

ROOT_PATH=$(realpath $(dirname $0)/..)
# needs to match dockerfile workdir
WORKDIR="/ball_vision"

DOCKER_ARGS="--rm -it --privileged --network=host "
DOCKER_ARGS+="-v $ROOT_PATH:$WORKDIR "

# Ensure DISPLAY is set
if [ -z "$DISPLAY" ]; then
  export DISPLAY=:0
fi

xhost +local:docker  # Allow Docker to access the X server
DOCKER_ARGS+="-e DISPLAY=$DISPLAY "
DOCKER_ARGS+="-v /tmp/.X11-unix:/tmp/.X11-unix "
DOCKER_ARGS+="--gpus all "  # Use all available GPUs
docker run \
  $DOCKER_ARGS \
  ball_image
