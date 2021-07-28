import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid

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
GLOBAL_PLAN_TIME = 0.5
LOCAL_PLAN_TIME = 0.05
PUBLISH_EVERY = 4

HEIGHT=480
WIDTH=480
PIXEL_PER_METER_X = (WIDTH - 2*150)/3.0 #Horizontal distance between src points in the real world ( I assumed 3.0 meters)
PIXEL_PER_METER_Y = (HEIGHT - 30-60)/8.0 #Vertical distance between src points in the real world ( I assumed 6.0 meters)

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision')
        # Publishers
        self.path_img_publisher_ = self.create_publisher(Image, 'vision/path_img', 1)
        self.birdeye_publisher_ = self.create_publisher(Image, 'vision/birdeye_img', 1)
        self.scan_publisher_ = self.create_publisher(LaserScan, 'segmentation/scan', 1)
        self.global_path_publisher_ = self.create_publisher(Path, 'global_plan', 1)
        self.cmd_publisher_ = self.create_publisher(Twist, 'cmd_vel', 1)
        self.costmap_publisher_ = self.create_publisher(OccupancyGrid, 'costmap', 1)

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
        self.publish_count_ = 0
        print('Vision Module ready!')


    def global_plan_callback(self):
        if self.image_msg_ is not None:
            image = self.bridge.imgmsg_to_cv2(self.image_msg_, desired_encoding='passthrough')
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Save a sample image
            if not self.is_saved:
                cv2.imwrite("/usr/src/app/dev_ws/src/vision/dolly.png", image)
                self.is_saved = True
            # image = cv2.resize(image,(960,720))

            # Perform one navigation step (perception and global path planning)
            global_plan, result_img, result_birdview = self.nav_.global_planner_step(image, self.robot_state_)
            self.publish_count_ += 1 
            if(global_plan.shape[0] > 1):
                self.global_plan_ = global_plan
                # Publish topics
                #Path
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
                # if(self.publish_count_ > PUBLISH_EVERY):
                self.global_path_publisher_.publish(path_msg)

                # Costmap
                costmap_msg = OccupancyGrid()
                costmap_msg.header.frame_id = 'odom_demo'
                costmap_msg.header.stamp = self.image_msg_.header.stamp
                costmap_msg.info.resolution = 4.0/PIXEL_PER_METER_X
                costmap = cv2.resize(result_birdview[:,:,0], (int(WIDTH/4.0),int((HEIGHT*PIXEL_PER_METER_X/PIXEL_PER_METER_Y)/4.0)))
                costmap = cv2.flip(costmap,-1)
                costmap_msg.info.width = costmap.shape[0]
                costmap_msg.info.height = costmap.shape[1]
                origin_costmap = poses[0].pose
                origin_costmap.orientation = self.robot_orientation_
                origin_costmap.position.y -= (WIDTH/(PIXEL_PER_METER_X*2.0))*math.cos(self.robot_state_[2])# Half of the image
                origin_costmap.position.x += (WIDTH/(PIXEL_PER_METER_X*2.0))*math.sin(self.robot_state_[2])# Half of the image
                # origin_costmap.orientation.w = 1.0
                costmap_msg.info.origin = origin_costmap
                costmap = np.transpose(costmap)
                costmap = costmap.flatten()
                costmap = np.int8(100.0*np.float32(costmap)/255.0)
                costmap_msg.data = costmap.tolist()
                self.costmap_publisher_.publish(costmap_msg)

                path_img = Image()
                    

                # result_img = cv2.cvtColor(np.uint8(result_img), cv2.COLOR_GRAY2RGB)

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
                self.publish_count_ = 0
            else:
                print("No global plan generated")

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
        self.robot_orientation_ = msg.pose.pose.orientation

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
