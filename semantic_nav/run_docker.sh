#!/bin/bash
sudo xhost +si:localuser:root
XSOCK=/tmp/.X11-unix

docker run -it --rm  \
    --net=host  \
    --privileged \
    --runtime=nvidia \
    -e DISPLAY=$DISPLAY \
    -v $XSOCK:$XSOCK \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v `pwd`/dev_ws/src:/usr/src/app/dev_ws/src \
    -v `pwd`/../pretrained_models:/usr/src/app/pretrained_models \
    amr "$@"