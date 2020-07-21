import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from fchardnet import SemanticSegmentation
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import os

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
        self.msg = Image()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Save a sample image
        if not self.is_saved:
            cv2.imwrite("/usr/src/app/dev_ws/src/semantic_segmentation/dolly.png", cv_image)
            self.is_saved = True
        # cv_image = cv2.resize(cv_image,(960,720))
        h,w,_ = cv_image.shape
        # Perform semantic segmentation
        img_decoded = self.seg.process_img(cv_image,[h,w])
        img_decoded = np.uint8(img_decoded * 255)
        #Warp
        warped = cv2.warpPerspective(img_decoded, self.M, self.img_size, flags=cv2.INTER_LINEAR)
        warped_msg = Image()
        warped_msg = self.bridge.cv2_to_imgmsg(warped)
        warped_msg.header.stamp = msg.header.stamp
        warped_msg.encoding = msg.encoding
        

        img_msg = self.bridge.cv2_to_imgmsg(img_decoded)
        self.msg.height = msg.height
        self.msg.width = msg.width
        self.msg.header.stamp = msg.header.stamp
        self.msg.encoding = msg.encoding
        self.seg_publisher_.publish(img_msg)
        self.warp_publisher_.publish(warped_msg)

def main(args = None):
    print('Hi from semantic_segmentation.')
    rclpy.init(args=args)
    semantic_segmentation = SemanticSegmentationNode()
    rclpy.spin(semantic_segmentation)

    semantic_segmentation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()