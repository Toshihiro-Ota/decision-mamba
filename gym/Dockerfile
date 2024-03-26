ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel as python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

FROM python-base as initial
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update && apt-get install -y curl git build-essential cmake ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN cp /etc/apt/sources.list /etc/apt/sources.list~ && sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list \
    && apt update && apt install -y gcc wget && apt build-dep -y mesa \
    && apt install -y llvm-dev freeglut3 freeglut3-dev build-essential python3-dev libgl1-mesa-dev libgl1-mesa-glx libglew-dev libglfw3-dev libosmesa6-dev patchelf \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && tar -zxvf mujoco210-linux-x86_64.tar.gz \
    && mkdir ~/.mujoco && cp -r mujoco210 ~/.mujoco/mujoco210 \
    && rm -r mujoco210 mujoco210-linux-x86_64.tar.gz


RUN conda clean --all
ENV FORCE_CUDA="1"

WORKDIR /workspace

FROM initial as development

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt --no-cache-dir

RUN sed -i -e '5d' /opt/conda/lib/python3.10/site-packages/mamba_ssm/__init__.py
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV D4RL_SUPPRESS_IMPORT_ERROR=1
