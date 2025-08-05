###############################################################################
# Base: PyTorch 1.10.2 - CUDA 11.3 - Ubuntu 20.04 (matches ULIP2 requirement) #
###############################################################################
FROM cnstark/pytorch:1.10.2-py3.9.12-cuda11.3.1-ubuntu20.04 

ARG DEBIAN_FRONTEND=noninteractive

#####################
# System packages   #
#####################
RUN sed -i.bak -e 's|http://archive.ubuntu.com/ubuntu|http://ftp.jaist.ac.jp/pub/Linux/ubuntu|g' \
               -e 's|http://security.ubuntu.com/ubuntu|http://ftp.jaist.ac.jp/pub/Linux/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git ninja-build build-essential cmake clang \
        libxi6 libxrender1 libxfixes3 libxxf86vm1 libx11-dev \
        mesa-utils x11-apps libglu1-mesa libopencv-dev python3-tk \
        wget libxml2 fonts-noto-cjk && \
    rm -rf /var/lib/apt/lists/*

#####################
# Blender 3.6.5     #
#####################
ENV BLENDER_VERSION=3.6.5
RUN wget -q https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}-linux-x64.tar.xz && \
    tar -xf blender-${BLENDER_VERSION}-linux-x64.tar.xz -C /opt && \
    ln -s /opt/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender && \
    rm blender-${BLENDER_VERSION}-linux-x64.tar.xz
ENV PATH="/opt/blender-${BLENDER_VERSION}-linux-x64:${PATH}"

#############################
# NVIDIA Container settings #
#############################
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

################################################################
# ----------  ULIP 2 (PyTorch side)  -------------------------- #
################################################################
WORKDIR /workspace/ulip2
COPY ulip2/requirements.txt .
RUN pip install -U pip && \
    pip install -r requirements.txt

# PointNet++ (needs a compile with nvcc inside the image)
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git && \
    pip install ./Pointnet2_PyTorch/pointnet2_ops_lib && \
    rm -rf Pointnet2_PyTorch

################################################################
# ----------  dataset_analyzer side  -------------------------- #
################################################################
WORKDIR /workspace/dataset_analyzer
COPY dataset_analyzer/requirements.txt .
RUN pip install -r requirements.txt
# (If this repo has extra build steps, add them here)

####################################
# Non-root user (optional, safer)  #
####################################
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /workspace

#####################
# Default entry     #
#####################
ENV PYTHONUNBUFFERED=1
CMD ["/bin/bash"]
