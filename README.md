# autonomous_mobile_robot
This is the implementation of an navigation system for an autonomous mobile robot based on:

1. Semantic Segmentation: From a front facing camera obects in the image are segmented and driveable areas identified.
2. Perspective Transform: A transformation matrix allows to transform the segmented image to a birdview perspective.
3. Scan Transformation: The contour of the driveable area is found and then distances from the camera to the driveable area contour are computed. This allows to generate a vector of distances, that resembles the output of a 2D Lidar (/scan msg type in ROS)
4. Dynamic Window Approach: DWA is used to compute a motion command (velocity and yaw rate) that keeps the robot inside the driveable areas while avoiding obstacles.

The simulation is implemented in gazebo and uses [dolly](https://github.com/chapulina/dolly) and [citysim](https://github.com/osrf/citysim) forks.

## Install
1. Install [ROS 2 eloquent](https://index.ros.org/doc/ros2/Installation/Eloquent/Linux-Install-Debians/).

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
    docker run -p 8888:8888 -v `pwd`/dev_ws/src/semantic_segmentation:/usr/src/app/dev_ws/src/semantic_segmentation -it --rm --gpus all amr 
    jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
    ```

2. Follow the instructions in the Jupytenotebook located inside the container in: */usr/src/app/dev_ws/src/semantic_segmentation/semantic_segmentation/warp_scan_segmentation.ipynb*

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
3. Run semantic segmentaton inside docker container
    ```bash
    cd dev_ws
    . /opt/ros/eloquent/setup.bash 
    colcon build
    . install/setup.bash
    export DOMAIN_ID=0
    run semantic_segmentation semantic_segmentation --ros-args --param use_sim_time:=true
    ```
4. In a new terminal run dwa_planner
    ```bash
    cd autonomous_mobile_robot
    . install/setup.bash
    ros2 run dwa_planner dwa_planner 
    ```
5. In a new terminal run the node that generates the waypoint of the path to be followed
    ```bash
    cd autonomous_mobile_robot
    . install/setup.bash
    ros2 run dwa_planner trajectory_publisher 
    ```

## Results
![alt text](imgs/results.gif "Title")

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
