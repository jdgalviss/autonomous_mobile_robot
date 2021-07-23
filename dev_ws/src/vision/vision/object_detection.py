import sys
sys.path.append("/usr/src/app/dev_ws/src/vision/vision")
import torch
import numpy as np
import cv2
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os


class ObjectDetector(object):
    def __init__(self, model_path):
        print('Loading Object Detection')
        self.model = attempt_load(model_path, map_location='cuda')  # load FP32 model
        self.colors = [(255,0,0),(0,255,0),(0,0,255), (255,255,255)]
        self.img_size = 640
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names


        self.conf_thres=0.25
        self.iou_thres=0.45
        self.max_det=100
        self.agnostic_nms=False

    def process_frame(self, image):
        img0 = image.copy()
        # img = img[:, :, ::-1] 
        img = letterbox(img0, self.img_size, stride=self.stride)[0] # Padded resize
        self.im_hw = img.shape
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cuda')
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        preds = self.model(img, augment=False)[0]
        preds = non_max_suppression(preds, self.conf_thres, self.iou_thres, None, self.agnostic_nms, max_det=self.max_det)
        preds = preds[0].cpu().numpy()
        return preds

    def draw_rectangles(self, img, preds):
        out = cv2.resize(img, (self.im_hw[1], self.im_hw[0]), interpolation = cv2.INTER_AREA)
        for p in preds:
            if(p[4] > 0.5):
                # print("row: {}".format(p))
                class_idx = min(int(p[5]), 3)
                color = self.colors[class_idx]
                out = cv2.rectangle(out, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, 2)
                out = cv2.putText(out, self.names[int(p[5])],  (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color, 1, cv2.LINE_AA)
        # out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        return out    
