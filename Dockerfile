FROM nvidiajetson/l4t-ros2-eloquent-pytorch:r32.5

RUN /bin/bash -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Install basic packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    git \
    unzip \
    curl \
    nano \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*


RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y python3-pip \
    ros-eloquent-cv-bridge \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*

COPY dev_ws/semantic_segmentation.sh /usr/src/app/dev_ws/
COPY requirements.txt /usr/src/app/
WORKDIR /usr/src/app
RUN pip3 install -r requirements.txt
WORKDIR /usr/src/app/dev_ws
