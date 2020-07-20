import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from fchardnet import SemanticSegmentation
import cv2
import numpy as np

class SemanticSegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')
        # Image subscriber
        self.image_sub_ = self.create_subscription(Image, 'camera/image_raw', self.image_raw_callback, 10)
        self.image_sub_  # prevent unused variable warning
        self.publisher_ = self.create_publisher(Image, 'vision/image_segmented', 10)
        self.bridge = CvBridge()
        self.seg = SemanticSegmentation()
        self.is_saved = False

    def image_raw_callback(self, msg):
        self.msg = Image()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        if not self.is_saved:
            cv2.imwrite("/usr/src/app/dev_ws/src/semantic_segmentation/dolly.png", cv_image)
            self.is_saved = True
        # cv_image = cv2.resize(cv_image,(960,720))
        h,w,_ = cv_image.shape

        img_decoded = self.seg.process_img(cv_image,[h,w])
        img_decoded = np.uint8(img_decoded * 255)
        # self.perform_inference(self.detection_model,cv_image)
        img_msg = self.bridge.cv2_to_imgmsg(img_decoded)
        self.msg.height = msg.height
        self.msg.width = msg.width
        self.msg.header.stamp = msg.header.stamp
        self.msg.encoding = msg.encoding
        self.publisher_.publish(img_msg)

def main(args = None):
    print('Hi from semantic_segmentation.')
    rclpy.init(args=args)
    semantic_segmentation = SemanticSegmentationNode()
    rclpy.spin(semantic_segmentation)

    semantic_segmentation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()