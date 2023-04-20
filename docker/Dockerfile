# How to setup
#
# You need NVidia driver 470. So run followings in host PC
#    sudo apt-get purge '*nvidia*'
#    sudo apt install nvidia-driver-470
#    sudo systemctl daemon-reload
#    sudo systemctl restart docker
#
# And you need to build your own pytorch 19.01
#     see how to : https://github.com/hamling-ling/PyTorchLocalBuild
#
# Then copy your pytorch whl in docker directory.
# And build docker image
# > docker build -f Dockerfile_tch \
#   --build-arg user=$(id -un) \
#   --build-arg user_id="$(id -u)" \
#   --build-arg user_grp="$(id -gn)" \
#   --build-arg user_gid="$(id -g)" \
#   --build-arg pass=1234 \
#   -t nlu-iot .
#
# Run container
# > docker run --rm -it nlu-iot
#

FROM nvcr.io/nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04

ENV TORCHWHL="torch-1.9.0a0+gitunknown-cp38-cp38-linux_x86_64.whl"

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND noninteractive

ARG user="jovyan"
ARG pass="jovyan"
ARG user_id=1000
ARG user_grp="jovyans"
ARG user_gid=1000

LABEL maintainer="Nobuhiro Kuroiwa"

RUN apt-get update && \
    apt-get install -y \
    python3-pip

RUN apt-get update && \
    apt-get install -y \
        curl \
        wget \
        zip \
        git \
        sudo \
        jupyter-notebook

COPY ${TORCHWHL} /tmp/
RUN pip install /tmp/${TORCHWHL}

RUN pip install seaborn keras
RUN pip install scikit-learn
RUN pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0  pytorch-lightning==1.6.1
RUN pip install ipywidgets==7.7.1
RUN pip install markupsafe==2.0.1
RUN pip install jupyter-tensorboard

RUN jupyter tensorboard enable --system
RUN jupyter nbextension enable --py widgetsnbextension

# User
RUN echo ${user}
RUN addgroup --gid ${user_gid} ${user_grp}
RUN useradd -u ${user_id} -g ${user_gid} -G sudo -s /bin/bash -d /home/${user} -m ${user}

# Debug
RUN apt-get install -y \
            sudo
RUN echo ${user}:${pass} | chpasswd
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo ${user}" ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${user}
WORKDIR /home/${user}

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.password=''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token=''" >> ~/.jupyter/jupyter_notebook_config.py 
RUN echo "c.NotebookApp.ip='0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py 
RUN echo "c.NotebookApp.open_browser=False" >> ~/.jupyter/jupyter_notebook_config.py 
RUN echo "c.NotebookApp.port=8888" >> ~/.jupyter/jupyter_notebook_config.py

