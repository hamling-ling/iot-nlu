# iot-nlu
Natural Japanese understanding experiment of IoT device control.

## Prerequisite

* Linux (Ubuntu >= 20.04)
* docker (>=23.0.3)

## Preparation

1. Install NVidia Driver 515
     ```
    sudo apt-get purge '*nvidia*'
    sudo apt install nvidia-driver-515
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```

2. Clone this repository under ~/Github

   If you clone anywhere else, you need to edit run.sh

   ```
   cd ~/
   mkdir Github
   cd Github
   git clone https://github.com/hamling-ling/iot-nlu.git
   ```

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

## Run container
   ```
    cd iot-nlu/docker
    ./run.sh
   ```

## Connect to jupyter notebook running in the container

1. Run jupyter notebook.
   ```
   cd iot-nlu/src
   jupyter notebook
   ```
2. Connect your client pc
   * http://[HOST_IP_ADDRESS]:8888/
