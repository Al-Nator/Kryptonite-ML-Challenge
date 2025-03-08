FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    wget \
    bash \
    git

RUN mkdir -p /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm /root/miniconda3/miniconda.sh

ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda init bash

SHELL ["/bin/bash", "-c"]

WORKDIR /app
COPY . /app

RUN conda env create -f environment.yml
RUN conda run -n krypto pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
RUN conda run -n krypto pip install numpy==2.2.3
RUN conda install -n krypto \
    https://anaconda.org/conda-forge/onnxruntime/1.20.1/download/linux-64/onnxruntime-1.20.1-py312h3875227_200_cuda.conda -y

EXPOSE 8000

CMD ["bash"]
