. /opt/ros/foxy/setup.sh
. /usr/share/gazebo/setup.sh
colcon build
. install/setup.sh
export DOMAIN_ID=1
ros2 launch dolly_gazebo dolly.launch.py world:=dolly_city.world