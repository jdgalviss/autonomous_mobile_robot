import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist



from cv_bridge import CvBridge, CvBridgeError
from ament_index_python.packages import get_package_share_directory
from navigation import NavigationSystem
from helpers import euler_from_quaternion
import cv2
import numpy as np
import os
import math
import time

pkg_sem_seg = get_package_share_directory('vision')
GLOBAL_PLAN_TIME = 1.0
LOCAL_PLAN_TIME = 0.1

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision')
        # Publishers
        self.path_img_publisher_ = self.create_publisher(Image, 'vision/path_img', 1)
        self.birdeye_publisher_ = self.create_publisher(Image, 'vision/birdeye_img', 1)
        self.scan_publisher_ = self.create_publisher(LaserScan, 'segmentation/scan', 1)
        self.global_path_publisher_ = self.create_publisher(Path, 'global_plan', 1)
        self.cmd_publisher_ = self.create_publisher(Twist, 'cmd_vel', 1)
        cmd = Twist()
        self.cmd_publisher_.publish(cmd)
        # Create Navigation System
        self.nav_ = NavigationSystem(False)

        # Robot State
        self.robot_state_ = np.array([0.0,0.0,0.0])

        self.bridge = CvBridge() # Bridge between opencv and img msgs

        # Subscribers
        self.image_sub_ = self.create_subscription(Image, 'camera/image_raw', self.image_raw_callback, 1)
        self.odom_sub_ = self.create_subscription(Odometry, 'odom', self.odom_callback, 1)

        


        # Planner timers
        self.global_plan_timer_ = self.create_timer(GLOBAL_PLAN_TIME, self.global_plan_callback)
        self.local_plan_timer_ = self.create_timer(LOCAL_PLAN_TIME, self.local_plan_callback)

        self.is_saved = False
        self.image_msg_ = None
        self.global_plan_ = None
        print('Vision Module ready!')


    def global_plan_callback(self):
        if self.image_msg_ is not None:
            image = self.bridge.imgmsg_to_cv2(self.image_msg_, desired_encoding='passthrough')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Save a sample image
            if not self.is_saved:
                cv2.imwrite("/usr/src/app/dev_ws/src/vision/dolly.png", image)
                self.is_saved = True
            # image = cv2.resize(image,(960,720))

            # Perform one navigation step (perception and global path planning)
            self.global_plan_, result_img, result_birdview = self.nav_.global_planner_step(image, self.robot_state_)
            # publish topics
            path_msg = Path()
            path_msg.header.frame_id = 'odom_demo'
            path_msg.header.stamp = self.image_msg_.header.stamp#self.get_clock().now().to_msg()
            poses = []
            for point in self.global_plan_:
                pose = PoseStamped()
                pose.header.frame_id = 'odom_demo'
                pose.header.stamp = self.image_msg_.header.stamp
                pose.pose.position.x = point[0]
                pose.pose.position.y = point[1]
                pose.pose.orientation.w = 1.0
                poses.append(pose)
            path_msg.poses = poses
            self.global_path_publisher_.publish(path_msg)

            path_img = Image()
            path_img = self.bridge.cv2_to_imgmsg(result_img)
            path_img.header.frame_id = 'chassis'
            path_img.header.stamp = self.get_clock().now().to_msg()
            path_img.encoding = self.image_msg_.encoding
            self.path_img_publisher_.publish(path_img)

            path_img = Image()
            path_img = self.bridge.cv2_to_imgmsg(result_birdview)
            path_img.header.frame_id = 'chassis'
            path_img.header.stamp = self.get_clock().now().to_msg()
            path_img.encoding = self.image_msg_.encoding
            self.birdeye_publisher_.publish(path_img)

    def local_plan_callback(self):
        if self.global_plan_ is not None:
            # Find point in global plan closest to robot
            vel_cmd,yaw_rate_cmd = self.nav_.local_planner_step(self.robot_state_, self.global_plan_)
            cmd_msg = Twist()
            cmd_msg.linear.x = vel_cmd 
            cmd_msg.angular.z = yaw_rate_cmd 
            self.cmd_publisher_.publish(cmd_msg)
        else:
            print("No global plan")




    def odom_callback(self, msg):
        # print("Odometry received!")

        # Obtain, x,y and yaw:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        _,_,yaw = euler_from_quaternion(msg.pose.pose.orientation)

        self.robot_state_ = np.array([x,y,yaw])
        # print(self.robot_state_)

    def image_raw_callback(self, msg):
        # print("New Image received!")
        self.image_msg_ = msg


def main(args = None):
    print('Hi from vision.')
    rclpy.init(args=args)
    vision = VisionNode()
    rclpy.spin(vision)
    print('what')
    vision.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
