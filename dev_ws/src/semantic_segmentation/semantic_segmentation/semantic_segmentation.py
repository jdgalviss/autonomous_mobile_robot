import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge, CvBridgeError
from fchardnet import SemanticSegmentation
from utils import segmented2scan
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import os
import math
import time


pkg_sem_seg = get_package_share_directory('semantic_segmentation')

class SemanticSegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')
        # Image subscriber
        self.image_sub_ = self.create_subscription(Image, 'camera/image_raw', self.image_raw_callback, 10)
        self.image_sub_  # prevent unused variable warning
        # Image publisher
        self.seg_publisher_ = self.create_publisher(Image, 'vision/image_segmented', 10)
        self.warp_publisher_ = self.create_publisher(Image, 'vision/image_segmented/warped', 10)
        self.scan_publisher_ = self.create_publisher(LaserScan, 'vision/image_segmented/scan', 10)


        self.bridge = CvBridge() # Bridge between opencv and img msgs
        # Load files
        model_path = os.path.join(pkg_sem_seg, 'pretrained', 'hardnet70_cityscapes_model.pkl')
        transform_mtx_path = os.path.join(pkg_sem_seg, 'pretrained', 'PerspectiveTransform.npz')
        mtxs = np.load(transform_mtx_path)
        self.M = mtxs['M']
        self.M_inv = mtxs['M_inv']
        self.seg = SemanticSegmentation(model_path)
        self.is_saved = False
        self.img_size = (480, 480)

    def image_raw_callback(self, msg):
        time1 = time.clock()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Save a sample image
        if not self.is_saved:
            cv2.imwrite("/usr/src/app/dev_ws/src/semantic_segmentation/dolly.png", cv_image)
            self.is_saved = True
        # cv_image = cv2.resize(cv_image,(960,720))
        h,w,_ = cv_image.shape
        # Perform semantic segmentation
        img_decoded, driveable_decoded = self.seg.process_img_driveable(cv_image,[h,w])
        img_decoded = np.uint8(img_decoded * 255)
        driveable_decoded = np.uint8(driveable_decoded * 255)

        #Warp
        warped = cv2.warpPerspective(driveable_decoded, self.M, self.img_size, flags=cv2.INTER_LINEAR)

        # generate scan_msg
        original_center = np.array([[[w/2,h]]],dtype=np.float32)
        warped_center = cv2.perspectiveTransform(original_center, self.M)[0][0]
        scan_distances, angle_increment = segmented2scan(warped, warped_center)
        # publish topics
        warped_msg = Image()
        warped_msg = self.bridge.cv2_to_imgmsg(warped)
        warped_msg.header.stamp = msg.header.stamp
        warped_msg.encoding = msg.encoding
        
        seg_img = Image()
        img_msg = self.bridge.cv2_to_imgmsg(img_decoded)
        seg_img.header.stamp = msg.header.stamp
        seg_img.encoding = msg.encoding

        scan_msg = LaserScan()

        scan_msg.ranges = scan_distances
        scan_msg.intensities = [0.0]*len(scan_distances)

        scan_msg.header.stamp = msg.header.stamp
        scan_msg.header.frame_id = 'chassis'

        scan_msg.angle_increment = angle_increment
        scan_msg.angle_max = 50*math.pi/180.0
        scan_msg.angle_min = -50*math.pi/180.0
        scan_msg.range_min = 1.0
        scan_msg.range_max = 20.0



        self.seg_publisher_.publish(img_msg)
        self.warp_publisher_.publish(warped_msg)
        if(len(scan_distances) > 50):
            self.scan_publisher_.publish(scan_msg)
        time2 = time.clock()
        print(time2-time1)
        

def main(args = None):
    print('Hi from semantic_segmentation.')
    rclpy.init(args=args)
    semantic_segmentation = SemanticSegmentationNode()
    rclpy.spin(semantic_segmentation)

    semantic_segmentation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()