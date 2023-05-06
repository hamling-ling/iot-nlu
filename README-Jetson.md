# iot-nlu inference for Jetson Orin Nano

This is an setup guide to run [iot-nlu](README.md "iot-nlu README") demo for Jetson Orin Nano.

## Setup JetPack 8.5.2

   Follow NVIDIA's [official guide]( https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit ) to install JetPack 8.5.2. The version must be exactly the same because of cuda compatibility. This project requires cuda 11.4.

## Install packages

   Basically what we need is things in [iot-nlu](docker/Dockerf'''ile.trt "Dockerfile for TensorRT"). However I had to do little more to make it work in Jetson Orin Nano. So please follow instructions below.

### Install apt modules
   ```
   sudo apt-get -y update
   sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev
   ```

### Set Python3 to default

   ```
   sudo rm -rf /usr/bin/python
   sudo ln -s /usr/bin/python3.8 /usr/bin/python
   ```

### Install python modules

    ```
    pip install protobuf==3.20.3
    pip install pyyaml==6
    pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.6.1
    pip install ipywidgets==7.7.1
    pip install markupsafe==2.0.1
    ```

### Make Mecab Work Correctly

    ```
    sudo ln -s $HOME/.local/lib/python3.8/site-packages/home/nobu/.local/lib/mecab/li
bmecab.so.2 /usr/lib/libmecab.so.2
    ```

### Install Jupyter Notebook (Option)

    If you want to run notebooks.

    ```
    sudo a-tget -y install jupyter-notebook
    echo "c.NotebookApp.password=''" >> ~/.jupyter/jupyter_notebook_config.py
    echo "c.NotebookApp.token=''" >> ~/.jupyter/jupyter_notebook_config.py 
    echo "c.NotebookApp.ip='0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py 
    echo "c.NotebookApp.open_browser=False" >> ~/.jupyter/jupyter_notebook_config.py 
    echo "c.NotebookApp.port=8888" >> ~/.jupyter/jupyter_notebook_config.py
    ```

### Install Onnx Runtime (Option)

     This is an option only for users who want to run onnx files.
     ```
     wget -c https://nvidia.box.com/shared/static/v59xkrnvederwewo2f1jtv6yurl92xso.whl -O onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
     pip install pip install onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
     ```

## Run Inference Demo

1. Clone repository
    ```
    git clone https://github.com/hamling-ling/iot-nlu.git
    ```
2. Download Engine File
    ```
    cd iot-nlu/src
    ./download.sh
    ```
3. Run python script

    ```
    python3 demo_trt.py
    ```
