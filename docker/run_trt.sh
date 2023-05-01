#!/bin/bash

function torch() {
    docker run --gpus all \
       -v $HOME/Github:$HOME/Github -e "HOME=$HOME" -e "GRANT_SUDO=yes"\
       --rm \
       -it \
       -p 8888:8888 \
       -p 6006:6006 \
       nlu-iot-trt bash
}

torch
