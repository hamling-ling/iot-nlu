#!/bin/bash

export TAG=model-0.0.1
export FILE=iot-nlu-model-0.0.1.tar.bz2
wget https://github.com/hamling-ling/iot-nlu/releases/download/$TAG/$FILE
tar jxvf $FILE
