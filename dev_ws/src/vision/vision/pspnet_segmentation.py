import sys
sys.path.append("/usr/src/app/dev_ws/src/vision/vision/semseg")
import os
import logging
import argparse
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from util import config
# from util.util import colorize

cv2.ocl.setUseOpenCL(False)

CONFIG_FILE = "semseg/config/ade20k/ade20k_pspnet50.yaml"

def get_parser():
    cfg = config.load_cfg_from_cfg_file(CONFIG_FILE)
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

class PSPNetSematicSegmentation(object):
    def __init__(self, config_file=CONFIG_FILE):
        # Load Parameters
        self.args_ =  config.load_cfg_from_cfg_file(config_file)
        self.logger_ = get_logger()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.args_.test_gpu)
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        self.mean_ = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        self.std_ = [item * value_scale for item in std]
        # self.colors_ = np.loadtxt(self.args_.colors_path).astype('uint8')
        self.label_colors_ = self.get_label_colors()

        # Load Model
        if self.args_.arch == 'psp':
            from model.pspnet import PSPNet
            self.model_ = PSPNet(layers=self.args_.layers, classes=self.args_.classes, zoom_factor=self.args_.zoom_factor, pretrained=False)
        elif self.args_.arch == 'psa':
            from model.psanet import PSANet
            self.model_ = PSANet(layers=self.args_.layers, classes=self.args_.classes, zoom_factor=self.args_.zoom_factor, compact=self.args_.compact,
                            shrink_factor=self.args_.shrink_factor, mask_h=self.args_.mask_h, mask_w=self.args_.mask_w,
                            normalization_factor=self.args_.normalization_factor, psa_softmax=self.args_.psa_softmax, pretrained=False)
        self.model_ = torch.nn.DataParallel(self.model_).cuda()
        cudnn.benchmark = True

        if os.path.isfile(self.args_.model_path):
            self.logger_ = get_logger().info("=> loading checkpoint '{}'".format(self.args_.model_path))
            checkpoint = torch.load(self.args_.model_path)
            self.model_.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger_ = get_logger().info("=> loaded checkpoint '{}'".format(self.args_.model_path))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(self.args_.model_path))


    def get_label_colors(self):

        colors = [  # [  0,   0,   0],
                    [128, 64, 128],
                    [244, 35, 232],
                    [70, 70, 70],
                    [102, 102, 156],
                    [190, 153, 153],
                    [153, 153, 153],
                    [250, 170, 30],
                    [220, 220, 0],
                    [107, 142, 35],
                    [152, 251, 152],
                    [0, 130, 180],
                    [220, 20, 60],
                    [255, 0, 0],
                    [0, 0, 142],
                    [0, 0, 70],
                    [0, 60, 100],
                    [0, 80, 100],
                    [0, 0, 230],
                    [119, 11, 32],
                ]
        return dict(zip(range(19), colors))


    def net_process(self, image, flip=True):
        input = torch.from_numpy(image.transpose((2, 0, 1))).float()
        if self.std_ is None:
            for t, m in zip(input, self.mean_):
                t.sub_(m)
        else:
            for t, m, s in zip(input, self.mean_, self.std_):
                t.sub_(m).div_(s)
        input = input.unsqueeze(0).cuda()
        if flip:
            input = torch.cat([input, input.flip(3)], 0)
        with torch.no_grad():
            output = self.model_(input)
        _, _, h_i, w_i = input.shape
        _, _, h_o, w_o = output.shape
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        if flip:
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        output = output.data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        return output

    def scale_process(self,model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
        ori_h, ori_w, _ = image.shape
        pad_h = max(crop_h - ori_h, 0)
        pad_w = max(crop_w - ori_w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
        new_h, new_w, _ = image.shape
        stride_h = int(np.ceil(crop_h*stride_rate))
        stride_w = int(np.ceil(crop_w*stride_rate))
        grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
        grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
        prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
        count_crop = np.zeros((new_h, new_w), dtype=float)
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                s_h = index_h * stride_h
                e_h = min(s_h + crop_h, new_h)
                s_h = e_h - crop_h
                s_w = index_w * stride_w
                e_w = min(s_w + crop_w, new_w)
                s_w = e_w - crop_w
                image_crop = image[s_h:e_h, s_w:e_w].copy()
                count_crop[s_h:e_h, s_w:e_w] += 1
                prediction_crop[s_h:e_h, s_w:e_w, :] += self.net_process(image_crop)
        prediction_crop /= np.expand_dims(count_crop, 2)
        prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
        prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return prediction


    def test(self, model, image, classes, base_size, crop_h, crop_w, scales):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += self.scale_process(model, image_scale, classes, crop_h, crop_w, h, w, self.mean_, self.std_)
        prediction = self.scale_process(model, image_scale, classes, crop_h, crop_w, h, w, self.mean_, self.std_)
        prediction = np.argmax(prediction, axis=2)
        gray = np.uint8(prediction)
        # color = colorize(gray, colors)
        return gray    

    def colorize(self,labels):
        r = labels.copy()
        g = labels.copy()
        b = labels.copy()
        for l in range(0, 19):
            r[labels == l] = self.label_colors_ [l][0]
            g[labels == l] = self.label_colors_ [l][1]
            b[labels == l] = self.label_colors_ [l][2]
        rgb = np.zeros((labels.shape[0], labels.shape[1], 3))
        rgb[:, :, 0] = r 
        rgb[:, :, 1] = g 
        rgb[:, :, 2] = b 
        return np.uint8(rgb)


    def process_img(self, img):
        decoded_img = self.test(self.model_.eval(),img,  self.args_.classes, self.args_.base_size, self.args_.test_h, self.args_.test_w, self.args_.scales)
        
        # decoded_img = np.array(decoded_img)
        # decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        return self.colorize(decoded_img)
