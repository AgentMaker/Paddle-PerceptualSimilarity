FROM nvidia/cuda:10.1-base-ubuntu16.04

LABEL maintainer="HighCWu <HighCWu@163.com>"

# This Dockerfile is forked from Tensorflow Dockerfile

# Pick up some PaddlePaddle gpu dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget && \
    MINICONDA="Miniconda3-py37_4.9.2-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA
ENV PATH /miniconda/bin:$PATH

# Install PaddlePaddle
RUN conda update -n base conda && \ 
    conda install paddlepaddle-gpu==2.0.2 cudatoolkit=10.1 -c paddle

# Install PerceptualSimilarity dependencies
RUN conda install numpy scipy jupyter matplotlib && \
    conda install -c conda-forge scikit-image && \
    apt-get install -y python-qt4 && \
    pip install opencv-python

# For CUDA profiling, TensorFlow requires CUPTI. Maybe PaddlePaddle needs this too.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# IPython
EXPOSE 8888

WORKDIR "/notebooks"

