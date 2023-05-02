# iot-nlu
Natural Japanese understanding experiment of IoT device control.

## Demo
   Here is a demonstration of Japanese intent classification and slot filling.

### Demo1

   Turn on the fan in kitchen.
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
   ```

### Demo2:

   Set 18 celsius degree to the airconditioner in living room.
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
      "type_id": "6 (温度計)"
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
   docker build -f ./Dockerfile \
   --build-arg user=$(id -un) \
   --build-arg user_id="$(id -u)" \
   --build-arg user_grp="$(id -gn)" \
   --build-arg user_gid="$(id -g)" \
   --build-arg pass=1234 \
   -t nlu-iot .
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
## Connect to jupyter notebook

1. Run jupyter notebook.

   In the container, type following commands to start jupyter notebook server.
   ```
   cd iot-nlu/src
   jupyter notebook
   ```

2. Connect your client pc

   In your client PC, type following URL in a browser to start.
   
   * http://[HOST_IP_ADDRESS]:8888/

