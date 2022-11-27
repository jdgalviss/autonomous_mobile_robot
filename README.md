# autonomous_mobile_robot
**Note:** Old Implementation using DWAin citysim: [dwa](https://github.com/jdgalviss/autonomous_mobile_robot/tree/dwa)

<!--
![01_results](https://user-images.githubusercontent.com/18732666/147863893-07543d57-ec36-4b0c-b735-990d4cc95fda.png)

<!--
![left0416](https://user-images.githubusercontent.com/18732666/150397275-75c7059a-a19b-430e-86b7-f61506884739.jpg)
-->

https://user-images.githubusercontent.com/18732666/147863969-dd330be5-d9da-4aa1-972c-b5d6edf766a6.mp4


This is the implementation of an navigation system for an autonomous mobile robot using only front-facing RGB Camera. The proposed approach uses **semantic segmentation** to detect drivable areas in an image and object detection to emphasize objects of interest such as people and cars using yolov5. These detections are then transformed into a Bird's-Eye view semantic map that also contains spatial information about the distance towards the edges of the drivable area and the objects around the robot. Then, a **multi-objective cost function** is computed from the semantic map and used to generate a safe path for the robot to follow. 

The code was tested on both simulation and a real robot (clearpath robotics' jackal).

The simulation is implemented in gazebo and uses [dolly](https://github.com/chapulina/dolly) and [citysim](https://github.com/osrf/citysim).

Semantic segmentation is strongly based on [PSPNet](https://github.com/hszhao/semseg) and [FCHardNet](https://github.com/PingoLH/FCHarDNet).

## Install
1. Install [ROS 2](https://index.ros.org/doc/ros2/Installation/Eloquent/Linux-Install-Debians/).

2. Install Docker following the instructions on the [link](https://docs.docker.com/engine/install/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (for gpu support). *Semantic segmentation will be run inside docker container, however it could be run on the host with the proper configuration of pytorch*.

3. Clone this repo and its submodules 
    ```bash
    git clone --recursive -j8 https://github.com/jdgalviss/autonomous_mobile_robot.git
    cd autonomous_mobile_robot
    ```
4. (Optional if you want tu use PSPNet, FCHardNet are already included) Download Semantic Segmentation pretrained models for PSPNet from the following link: [Google Drive](https://drive.google.com/drive/folders/1pwOLNTVaKQVt4uSUl7ynOKvLA0_Qk4Rc). This is the required folder structure for these models:
    ```
    autonomous_mobile_robot
    |   ...
    └───pretrained_models
        |   ...
        └───exp
            └───ade20k
            |   |   ...
            |
            └───cityscapes
            |   |   ...
            |
            └───voc2012
                |   ...
    ```

4. Build Dockerfile.
    ```bash
    cd semantic_nav
    docker build . -t amr
    ```
## Run
1. Run docker container using provided script
    ```bash
    source run_docker.sh
    ```

### Run Simulation
1. Inside the docker container, run ros2/gazebo simulation using the provided scripts (The first time, it might take a few minutes for gazebo to load all the models)
    ```bash
    source run.sh
    ```
### Test Semantic Segmentation and calculate perspective transformation matrix

1. Run docker container and jupyterlab
    ```bash
    source run_docker.sh

    jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
    ```

2. Follow the instructions in the Jupytenotebook located inside the container in: */usr/src/app/dev_ws/src/vision/vision/_calculate_perspective_transform.ipynb*

Additional notebooks are provided in */usr/src/app/dev_ws/src/vision/vision/* to explain some of the concepts used in this work.
    
<!--
## Results





Semantic Segmentation + Perspective Transformation
![01_perspective](https://user-images.githubusercontent.com/18732666/147863902-18efad81-1d0e-4b3f-8b61-916a744fb96f.png)

Multi-objective Cost
![semanticNav_cost](https://user-images.githubusercontent.com/18732666/147863922-71ebb1f9-97b8-4ef0-9ec4-24e04077310b.png)

-->

<!-- # Launch Doly
cd dev_ws
. /opt/ros/eloquent/setup.bash 
. /usr/share/gazebo/setup.sh
. /usr/local/share/citysim/setup.sh
. install/setup.bash
export DOMAIN_ID=0
ros2 launch dolly_gazebo dolly.launch.py world:=simple_city_orig.world

# Launch semantic
cd dev_ws
. /opt/ros/eloquent/setup.bash 
colcon build
. install/setup.bash
export DOMAIN_ID=0

ros2 run semantic_segmentation semantic_segmentation  -->
