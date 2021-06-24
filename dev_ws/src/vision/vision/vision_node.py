import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge, CvBridgeError
from segmentation import SemanticSegmentation
from utils import warped2scan
from ament_index_python.packages import get_package_share_directory
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
        # Image publisher
        self.seg_publisher_ = self.create_publisher(Image, 'vision/image', 1)
        self.seg_raw_publisher_ = self.create_publisher(Image, 'vision/segmented_image_raw', 1)
        self.scan_publisher_ = self.create_publisher(LaserScan, 'segmentation/scan', 1)
        self.bridge = CvBridge() # Bridge between opencv and img msgs

        # Load files
        segmentation_model_path = os.path.join(pkg_sem_seg, 'pretrained', 'hardnet70_cityscapes_model.pkl')
        self.seg_model_ = SemanticSegmentation(segmentation_model_path)
        transform_mtx_path = os.path.join(pkg_sem_seg, 'pretrained', 'PerspectiveTransform.npz')
        mtxs = np.load(transform_mtx_path)
        self.M = mtxs['M']
        self.M_inv = mtxs['M_inv']

        self.is_saved = False

    def seg2scan(self, driveable_area):
        h,w,_ = driveable_area.shape
        warped = cv2.warpPerspective(driveable_area, self.M, (480, 480), flags=cv2.INTER_LINEAR)
        original_center = np.array([[[w/2,h]]],dtype=np.float32)
        warped_center = cv2.perspectiveTransform(original_center, self.M)[0][0]
        scan_distances, angle_increment, warped_contours = warped2scan(warped, warped_center)
        return warped, warped_contours, scan_distances, angle_increment

    def image_raw_callback(self, msg):
        print("New Image received!")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Save a sample image
        if not self.is_saved:
            cv2.imwrite("/usr/src/app/dev_ws/src/semantic_segmentation/dolly.png", cv_image)
            self.is_saved = True
        # cv_image = cv2.resize(cv_image,(960,720))
        # Perform semantic segmentation
        start = time.time()
        h,w,_ = cv_image.shape
        img_decoded, driveable_decoded = self.seg_model_.process_img_driveable(cv_image,[h,w])
        print("Semantic Seg time: {} seconds".format(time.time()-start))
        # Compute 2D Scan and image
        start = time.time()
        warped_original, warped_contours, scan_distances, angle_increment = self.seg2scan(driveable_decoded)
        print("Scan msg generation and warp: {} seconds".format(time.time()-start))
        # Unwarp ang generate output image
        start = time.time()
        unwarped_driveable = cv2.warpPerspective(warped_contours, self.M_inv, (w,h), flags=cv2.INTER_LINEAR)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        out_driveable = cv2.addWeighted(cv_image, 0.7, unwarped_driveable, 0.3, 0)
        print("Visual generation: {} seconds".format(time.time()-start))
        # publish topics
        seg_img = Image()
        seg_img = self.bridge.cv2_to_imgmsg(out_driveable)
        seg_img.header.frame_id = 'chassis'
        seg_img.header.stamp = self.get_clock().now().to_msg()
        seg_img.encoding = msg.encoding
        self.seg_publisher_.publish(seg_img)

        seg_img = Image()
        seg_img = self.bridge.cv2_to_imgmsg(warped_original)
        seg_img.header.frame_id = 'chassis'
        seg_img.header.stamp = self.get_clock().now().to_msg()
        seg_img.encoding = msg.encoding
        self.seg_raw_publisher_.publish(seg_img)

        scan_msg = LaserScan()
        scan_msg.ranges = scan_distances
        scan_msg.intensities = [0.0]*len(scan_distances)
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'chassis'
        scan_msg.angle_increment = angle_increment
        scan_msg.angle_max = 50*math.pi/180.0
        scan_msg.angle_min = -50*math.pi/180.0
        scan_msg.range_min = 1.0
        scan_msg.range_max = 20.0
        if(len(scan_distances) > 50):
            self.scan_publisher_.publish(scan_msg)

def main(args = None):
    print('Hi from vision.')
    rclpy.init(args=args)
    vision = VisionNode()
    rclpy.spin(vision)

    vision.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
