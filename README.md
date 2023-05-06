# iot-nlu

The goal of this project is to enable seamless Japanese language processing for edge computing, allowing for efficient control of IoT devices.

## Demo
   Here is a demonstration of intent classification and slot filling for Japanese language.

### Demo1

   Turn on the fan in the kitchen.
   ```
   input> キッチンの換気扇をつけて
   took 8.902999999999999 ms
   Input   : キッチンの換気扇をつけて
   Intent  : 1 (点灯したい)
   Entities: [
   {
      "name": "キッチン",
      "type_id": "3 (設置場所)"
   },
   {
      "name": "換気扇",
      "type_id": "4 (オンオフできる物)"
   }
   ]
   ```

### Demo2:

   Set the air conditioner in the living room to 18 degrees Celsius.
   ```
   input > 居間のエアコンを18度に設定して
   took 8.906 ms
   Input   : 居間のエアコンを18度に設定して
   Intent  : 6 (数値設定したい)
   Entities: [
   {
      "name": "居間",
      "type_id": "3 (設置場所)"
   },
   {
      "name": "エアコン",
      "type_id": "6 (温度調節できる物)"
   },
   {
      "name": "18",
      "type_id": "7 (温度)"
   }
   ]
   ```

## Prerequisite

* Linux (Ubuntu >= 20.04)
* docker (>=23.0.3)
* NVIDIA GPU (Tested on RTX3060 12GB)

## Preparation

1. Install CUDA 11.7 capable driver
   
   Pytorch 2.0.0 requires CUDA 11.7. So we need to install driver version 515.
   ```
    sudo apt-get purge '*nvidia*'
    sudo apt install nvidia-driver-515
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```

2. Clone this repository

   We supposed you have Github directory under ~/ and close repository in there. If you clone anywhere else, you need to edit run.sh and run_trt.sh

   ```
   cd ~/
   mkdir Github
   cd Github
   git clone https://github.com/hamling-ling/iot-nlu.git
   ```

# Containers

   This project includes 2 Dockerfiles for different notebook.
   Most of notebooks can run in a container derived from Dockerfile (we tag nlu-iot later). However only TensorRT specific notebook can run in a container built with Dockerfile.trt (we tag nlu-iot-trt later).

   Here is a table shows what script can run in which container.
   |Notebook           |Dockerfile     |Description                                |
   |-------------------|---------------|-------------------------------------------|
   |finetune.ipynb     |Dockerfile     |Finetune Joint-BERT and save model as ckpt.|
   |inference.ipynb    |Dockerfile     |Load ckpt and perform inference.           |
   |quantize.ipynb     |Dockerfile     |Load ckpt and perform dynamic quantization |
   |conv2onnx.ipynb    |Dockerfile     |Convert ckpt to onnx.                      |
   |conv2trt.ipynb     |Dockerfile.trt |Convert onnx to TRT.                       |
   |inference_trt.ipynb|Dockerfile.trt |Run TRT engine.                            |

## Build container
   
   Execute following commands to build container.
   ```
   cd iot-nlu/docker
   docker build -f ./Dockerfile \
   --build-arg user=$(id -un) \
   --build-arg user_id="$(id -u)" \
   --build-arg user_grp="$(id -gn)" \
   --build-arg user_gid="$(id -g)" \
   --build-arg pass=1234 \
   -t nlu-iot .
   ```

   And following commans to build container for TensorRT specific notebooks.
   ```
   cd iot-nlu/docker
   docker build -f ./Dockerfile.trt \
   --build-arg user=$(id -un) \
   --build-arg user_id="$(id -u)" \
   --build-arg user_grp="$(id -gn)" \
   --build-arg user_gid="$(id -g)" \
   --build-arg pass=1234 \
   -t nlu-iot-trt .
   ```

## Run container

   To run iot-nlu
   ```
    cd iot-nlu/docker
    ./run.sh
   ```

   Or to run iot-nlu-trt

   ```
    cd iot-nlu/docker
    ./run_trt.sh
   ```

# Run Demo Program

   1. Download model files

   ```
    cd iot-nlu
    ./download.sh
   ```
   
   2. Run demo script

      To run PyTorch demo, run the iot-nlu container and ...
      ```
      cd iot-nlu/src
      python3 demo.py
      ```

      And to run TensorRT demo, run the iot-nlu container and ...
      Note that engine file downloaded by the download.sh is built for Jetson Orin Nano. So for other environment, please run conv2onnx.ipynb then conv2trt.ipynb to buid engine file for your own enviroment.

      ```
      cd iot-nlu/src
      python3 demo_trt.py
      ```

## Run notebooks

1. Run jupyter notebook.

   In the container, type following commands to start jupyter notebook server.
   ```
   cd iot-nlu/src
   jupyter notebook
   ```

2. Connect your client pc

   In your client PC, type following URL in a browser to start.
   
   * http://[HOST_IP_ADDRESS]:8888/

