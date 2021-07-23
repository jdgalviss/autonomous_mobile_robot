import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


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


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision')
        # Image subscriber
        self.image_sub_ = self.create_subscription(Image, 'camera/image_raw', self.image_raw_callback, 1)
        self.odom_sub_ = self.create_subscription(Odometry, 'odom', self.odom_callback, 1)

        # Image publisher
        self.path_img_publisher_ = self.create_publisher(Image, 'vision/path_img', 1)
        self.birdeye_publisher_ = self.create_publisher(Image, 'vision/birdeye_img', 1)
        self.scan_publisher_ = self.create_publisher(LaserScan, 'segmentation/scan', 1)
        self.global_path_publisher_ = self.create_publisher(Path, 'global_plan', 1)

        self.bridge = CvBridge() # Bridge between opencv and img msgs

        # Create Navigation System
        self.nav_ = NavigationSystem(False)

        # Robot State
        self.robot_state_ = np.array([0.0,0.0,0.0])

        self.is_saved = False
        print('Vision Module ready!')


    def odom_callback(self, msg):
        print("Odometry received!")

        # Obtain, x,y and yaw:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        _,_,yaw = euler_from_quaternion(msg.pose.pose.orientation)

        self.robot_state_ = np.array([x,y,yaw])
        # print(self.robot_state_)

        

    def image_raw_callback(self, msg):
        # print("New Image received!")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Save a sample image
        if not self.is_saved:
            cv2.imwrite("/usr/src/app/dev_ws/src/vision/dolly.png", cv_image)
            self.is_saved = True
        # cv_image = cv2.resize(cv_image,(960,720))

        # Perform one navigation step (perception and path planning)
        path, result_img, result_birdview = self.nav_.path_planning(cv_image, self.robot_state_)

        # publish topics
        path_msg = Path()
        path_msg.header.frame_id = 'odom_demo'
        path_msg.header.stamp = msg.header.stamp#self.get_clock().now().to_msg()
        poses = []
        for point in path:
            pose = PoseStamped()
            pose.header.frame_id = 'odom_demo'
            pose.header.stamp = msg.header.stamp
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
        path_img.encoding = msg.encoding
        self.path_img_publisher_.publish(path_img)

        path_img = Image()
        path_img = self.bridge.cv2_to_imgmsg(result_birdview)
        path_img.header.frame_id = 'chassis'
        path_img.header.stamp = self.get_clock().now().to_msg()
        path_img.encoding = msg.encoding
        self.birdeye_publisher_.publish(path_img)

        


        # scan_msg = LaserScan()
        # scan_msg.ranges = scan_distances
        # scan_msg.intensities = [0.0]*len(scan_distances)
        # scan_msg.header.stamp = self.get_clock().now().to_msg()
        # scan_msg.header.frame_id = 'chassis'
        # scan_msg.angle_increment = angle_increment
        # scan_msg.angle_max = 50*math.pi/180.0
        # scan_msg.angle_min = -50*math.pi/180.0
        # scan_msg.range_min = 1.0
        # scan_msg.range_max = 20.0
        # if(len(scan_distances) > 50):
        #     self.scan_publisher_.publish(scan_msg)

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
