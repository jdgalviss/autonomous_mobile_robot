. /opt/ros/foxy/setup.sh
colcon build
. install/setup.bash
export DOMAIN_ID=1
#ros2 run semantic_segmentation semantic_segmentation --ros-args --param use_sim_time:=true
ros2 run vision vision_node --ros-args --param use_sim_time:=true
