# autonomous_mobile_robot
This is the implementation of an navigation system for an autonomous mobile robot based on:

1. Semantic Segmentation: From a front facing camera obects in the image are segmented and driveable areas identified.
2. Perspective Transform: A transformation matrix allows to transform the segmented image to a birdview perspective.
3. Scan Transformation: The contour of the driveable area is found and then distances from the camera to the driveable area contour are computed. This allows to generate a vector of distances, that resembles the output of a 2D Lidar (/scan msg type in ROS)
4. Dynamic Window Approach: DWA is used to compute a motion command (velocity and yaw rate) that keeps the robot inside the driveable areas while avoiding obstacles.

## Install
1. Install [ROS 2 dashing](https://index.ros.org/doc/ros2/Installation/Dashing/Linux-Install-Debians/).

2. Install Docker following the instructions on the [link](https://docs.docker.com/engine/install/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (for gpu support)..

3, 
```bash
docker build . -t amr
```
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

docker run -p 8888:8888 -v `pwd`/dev_ws/src/semantic_segmentation:/usr/src/app/dev_ws/src/semantic_segmentation -it --rm --gpus all amr 

## First shell
Run docker
. /opt/ros/dashing/setup.bash
. /opt/ros/melodic/setup.bash
export ROS_MASTER_URI=http://192.168.0.189:11311
export DOMAIN_ID=0
ros2 run ros1_bridge dynamic_bridge --bridge-all-1to2-topics

##Second shell
 . /usr/local/share/citysim/setup.sh
. catkin_ws/devel/setup.bash
roslaunch skid_steer_bot run_world.launch

##Third shell
docker ps
docker exec
cd dev_ws
. /opt/ros/dashing/setup.bash 
colcon build
cd ..
. dev_ws/install/setup.bash
export DOMAIN_ID=0

/opt/conda/lib/python3.7
/opt/conda/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.7:/opt/conda/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.7/site-packages/torch/


# Launch Doly
cd dev_ws
. /opt/ros/dashing/setup.bash 
. /usr/share/gazebo/setup.sh
. /usr/local/share/citysim/setup.sh
. install/setup.bash
export DOMAIN_ID=0
ros2 launch dolly_gazebo dolly.launch.py world:=simple_city_orig.world

# Launch semantic
cd dev_ws
. /opt/ros/dashing/setup.bash 
colcon build
. install/setup.bash
export DOMAIN_ID=0

ros2 run semantic_segmentation semantic_segmentation 
