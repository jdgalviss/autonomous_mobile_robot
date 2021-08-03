import cv2
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import os
from object_detection import ObjectDetector
from segmentation import SemanticSegmentation
from helpers import get_driveable_mask2
import matplotlib.pyplot as plt

class PerceptionSystem(object):
    def __init__(self, debug = False):
        self.debug_ = debug
        # Load Models
        segmentation_model_path = os.path.join('/usr/src/app/dev_ws/src/vision/vision', 'pretrained', 'hardnet70_cityscapes_model.pkl')
        self.seg_model_ = SemanticSegmentation(segmentation_model_path)
        object_detection_model_path = os.path.join('/usr/src/app/dev_ws/src/vision/vision', 'pretrained', 'yolov5s.pt')
        self.object_detector_ = ObjectDetector(object_detection_model_path)
        
        # Load perspective transforms
        mtxs = np.load('/usr/src/app/dev_ws/src/vision/vision/PerspectiveTransform.npz')
        self.M_ = mtxs['M']
        self.M_inv_ = mtxs['M_inv']
        
        # Test Detection Models
        print('Segmentation and Detection Models loaded, Testing the models')
        img = cv2.imread("/usr/src/app/dev_ws/src/vision/vision/data/73.png")
        self.h_orig_, self.w_orig_,_ = img.shape
        _, _ = self.seg_model_.process_img_driveable(img,[self.h_orig_,self.w_orig_])
        _ = self.object_detector_.process_frame(img)
        self.im_hw_ = self.object_detector_.im_hw
        print('Imgs tested')
        
    def get_driveable(self, driveable_decoded):
        h,w,_ = driveable_decoded.shape
        # Warp driveable area
        warped = cv2.warpPerspective(driveable_decoded, self.M_, (480, 480), flags=cv2.INTER_LINEAR)
        if(self.debug_):
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(warped)
            plt.show()
        # Calculate robot center
        original_center = np.array([[[w/2,h]]],dtype=np.float32)
        warped_center = cv2.perspectiveTransform(original_center, self.M_)[0][0]   
        driveable_contour_mask = get_driveable_mask2(warped, warped_center)
        return driveable_contour_mask
    
    def add_detections_birdview(self, preds, driveable_mask):
        h,w,_ = self.object_detector_.im_hw
        h_rate = self.h_orig_/h
        w_rate = self.w_orig_/w
        for pred in preds:
            if(pred[4] > self.object_detector_.conf_thres): # If prediction has a bigger confidence than the threshold
                x = w_rate*(pred[0]+pred[2])/2.0 # Ground middle point
                y = h_rate*pred[3]
                if(pred[5]==0): #person
                    wr = 40
                    hr = 60
                    color = 253#253
                else:
                    wr = 30
                    hr = 90
                    color = 255#255
                pos_orig = np.array([[[x,y]]],dtype=np.float32)
                warped_birdview = cv2.perspectiveTransform(pos_orig, self.M_)[0][0] # Transform middle ground point to birdview
                warped_birdview = np.uint16(warped_birdview)
                cv2.rectangle(driveable_mask, (warped_birdview[0] -int(wr/2), warped_birdview[1]), (warped_birdview[0] +int(wr/2), warped_birdview[1]-hr), color, -1) 
        
    def process_frame(self,img):
        # Semantic Segmentation
        img_test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_decoded, driveable_decoded = self.seg_model_.process_img_driveable(img_test,[self.h_orig_,self.w_orig_])
        
        # Get bird eye view with driveable area limits
        driveable_mask  = self.get_driveable(driveable_decoded)
        
        # Object Detection
        preds = self.object_detector_.process_frame(img)
        if(self.debug_):
            detections_img = self.object_detector_.draw_rectangles(img, preds)
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(detections_img)
            plt.show()
        
        # Add Detections to birdview image
        driveable_mask_with_objects = driveable_mask.copy()
        self.add_detections_birdview(preds, driveable_mask_with_objects)
        
        return driveable_decoded, driveable_mask, preds, driveable_mask_with_objects, img_decoded