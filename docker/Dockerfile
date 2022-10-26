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

RUN /bin/bash -c "source activate detectron2 && pip install opencv-python && \
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html"
    
SHELL ["conda", "run", "--no-capture-output", "-n", "detectron2", "/bin/bash", "-c"]
