. /opt/ros/eloquent/setup.sh
. /usr/share/gazebo/setup.sh
colcon build
. install/setup.sh
export DOMAIN_ID=0
ros2 launch dolly_gazebo dolly.launch.py world:=dolly_city.world

