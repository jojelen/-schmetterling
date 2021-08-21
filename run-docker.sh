#!/bin/bash

xhost +
docker run --volume $PWD:/piplinjen \
           --device=/dev/video0 \
           --volume /tmp/.X11-unix:/tmp/.X11-unix \
           -v /dev:/dev \
           --env DISPLAY=$DISPLAY \
           --privileged \
           --rm -it piplinjen
