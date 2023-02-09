FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx && apt-get -y install libglib2.0-0 && apt-get -y install nano && apt-get -y install vim
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/* 

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /workspace/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda --version

RUN conda create -n detectron2 python=3.8.11 \
    && conda init bash \
    && echo "conda activate detectron2" >> ~/.bashrc

RUN mkdir -p /workspace/output && mkdir -p /workspace/dataset && mkdir -p /workspace/src && mkdir -p /workspace/models 
ADD detectron2 /workspace/detectron2/

RUN /bin/bash -c "source activate detectron2 && pip install opencv-python && \
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    python -m pip install -e /workspace/detectron2 && pip install tensorpack==0.11 \
    pip install docker && pip install fairscale && pip install timm && pip install tritonclient[all]"

RUN adduser --disabled-password --gecos "" tbelldev

SHELL ["conda", "run", "--no-capture-output", "-n", "detectron2", "/bin/bash", "-c"]