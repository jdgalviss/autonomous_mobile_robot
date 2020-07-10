import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from fchardnet import SemanticSegmentation

class SemanticSegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')
        # Image subscriber
        self.image_sub_ = self.create_subscription(Image, 'robot/camera/image_raw', self.image_raw_callback, 10)
        self.image_sub_  # prevent unused variable warning
        self.publisher_ = self.create_publisher(Image, 'robot/image_segmented', 10)
        self.bridge = CvBridge()

    def image_raw_callback(self, msg):
        print("receiving raw image")
        self.msg = Image()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        #img_decoded = seg.process_img(cv_image,[msg.height,msg.width])
        # self.perform_inference(self.detection_model,cv_image)
        img_msg = self.bridge.cv2_to_imgmsg(cv_image)
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