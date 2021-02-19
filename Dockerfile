FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

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

# Install python dependencies
COPY requirements.txt .
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y python3-setuptools \
    python3-pip \
    && pip3 install --upgrade pip \
    && pip3 install -r requirements.txt \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 uninstall -y ipython
RUN pip3 install ipython
RUN pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Install ROS 2
RUN apt-get update && apt-get install -y locales
RUN /bin/bash -c "locale-gen en_US en_US.UTF-8; update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8"
ENV LANG=en_US.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
RUN DEBIAN_FRONTEND=noninteractive apt update && \
    apt install -y gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*
RUN echo "Installing ros dashing" 
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN /bin/bash -c 'echo "deb [arch=amd64,arm64] http://packages.ros.org/ros2/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list'
RUN apt update && \
    apt install -y ros-dashing-desktop\
    && rm -rf /var/lib/apt/lists/*
RUN /bin/bash -c "source /opt/ros/dashing/setup.bash"
RUN DEBIAN_FRONTEND=noninteractive apt update && \
    apt install -y python3-argcomplete \
    python3-rosdep \
    python3-colcon-common-extensions \
    nano \
    && rm -rf /var/lib/apt/lists/*
RUN /bin/bash -c "rosdep init; rosdep update"
ENV ROS_DOMAIN_ID=0

# Install ROS 1
RUN /bin/bash -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN DEBIAN_FRONTEND=noninteractive apt update && \
    apt install -y ros-melodic-ros-base \
    libeigen3-dev \
    ros-melodic-catkin \
    python-catkin-tools \
    ros-dashing-ros1-bridge \
    && rm -rf /var/lib/apt/lists/*

#COPY ros1_bridge/src/semantic_segmentation /usr/src/app/ros1_bridge/src/semantic_segmentation

WORKDIR /usr/src/app/