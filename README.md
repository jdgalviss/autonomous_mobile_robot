# autonomous_mobile_robot
**Note:** Implementation in progress. For a stable version check branch [dwa](https://github.com/jdgalviss/autonomous_mobile_robot/tree/dwa)

This is the implementation of an navigation system for an autonomous mobile robot using only front-facing RGB Camera. The proposed approach uses **semantic segmentation** to detect drivable areas in an image. These detections are then transformed into a Bird's-Eye view semantic map that also contains spatial information about the distance towards the edges of the drivable area and the objects around the robot. Then, a **multi-objective cost function** is computed from the semantic map and used to generate a safe path for the robot to follow. 

The code was tested on both simulation and a real robot (clearpath robotics' jackal).

The simulation is implemented in gazebo and uses [dolly](https://github.com/chapulina/dolly) and [citysim](https://github.com/osrf/citysim).

## Install
1. Install [ROS 2](https://index.ros.org/doc/ros2/Installation/Eloquent/Linux-Install-Debians/).

2. Install Docker following the instructions on the [link](https://docs.docker.com/engine/install/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (for gpu support). *Semantic segmentation will be run inside docker container, however it could be run on the host with the proper configuration of pytorch*.

3. Clone this repo and its submodules 
    ```bash
    git clone --recursive -j8 https://github.com/jdgalviss/autonomous_mobile_robot.git
    cd autonomous_mobile_robot
    ```

4. Build citysim
    ```bash
    cd citysim
    mkdir build
    cd build
    cmake ..
    make install
    ```

5. Build Dockerfile.
    ```bash
    docker build . -t amr
    ```

6. Build ros workspace
    ```bash
    cd dev_ws
    colcon build
    ```
## Run

### Test Semantic Segmentation and calculate perspective transformation matrix

1. Run docker container and jupyterlab
    ```bash
    source run_docker.sh

    jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
    ```

2. Follow the instructions in the Jupytenotebook located inside the container in: */usr/src/app/dev_ws/src/vision/vision/_calculate_perspective_transform.ipynb*

### Run Simulation
1. On a new terminal run the simulation:
    ```bash
    cd dev_ws
    . /opt/ros/eloquent/setup.bash 
    . install/setup.bash
    . /usr/local/share/citysim/setup.sh
    export DOMAIN_ID=0
    ros2 launch dolly_gazebo dolly.launch.py world:=simple_city_orig.world
    ```
2. In another terminal, run docker container
    ```bash
    cd autonomous_mobile_robot
    docker run -p 8888:8888 -v `pwd`/dev_ws/src/semantic_segmentation:/usr/src/app/dev_ws/src/semantic_segmentation -it --rm --gpus all amr 
    ```
3. Run docker container and then the semantic segmentation node
    ```bash
    source run_docker.sh
    source semantic_segmentation_nav.sh
    ```
    

## Results


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
