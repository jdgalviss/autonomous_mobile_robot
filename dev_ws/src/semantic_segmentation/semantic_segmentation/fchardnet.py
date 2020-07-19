import sys
import torch
from utils import convert_state_dict
import cv2
import numpy as np
sys.path.append("/usr/src/app/dev_ws/src/semantic_segmentation/semantic_segmentation")
from models import get_model


class SemanticSegmentation(object):
    def __init__(self):
        print("Semantic Segmentation using FCHardNet")
        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model("/usr/src/app/dev_ws/src/semantic_segmentation/semantic_segmentation/pretrained/hardnet70_cityscapes_model.pkl").to(self.device)


    def init_model(self, model_path):
        n_classes = 19
        # Setup Model
        model = get_model({"arch": "hardnet"}, n_classes)
        state = convert_state_dict(torch.load(model_path, map_location=self.device)["model_state"])
        model.load_state_dict(state)
        model.eval()
        model.to(self.device)
        return model

    def decode_segmap(self, temp,label_colours):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, 19):
            r[temp == l] = label_colours[l][0]
            g[temp == l] = label_colours[l][1]
            b[temp == l] = label_colours[l][2]
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def process_img(self, img, size, is_driveable_area = False):
        #print("Read Input Image from : {}".format(img_path))

        img_resized = cv2.resize(img, (int(size[1]), int(size[0])))  # uint8 with RGB mode
        img = img_resized.astype(np.float16)

        # norm
        value_scale = 255
        mean = [0.406, 0.456, 0.485]
        mean = [item * value_scale for item in mean]
        std = [0.225, 0.224, 0.229]
        std = [item * value_scale for item in std]
        img = (img - mean) / std

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()

        images = img.to(self.device)
        outputs = self.model(images)
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
        colors2 = [  # [  0,   0,   0],
            [125, 125, 125],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        if is_driveable_area:
            label_colours = dict(zip(range(19), colors2))
        else:
            label_colours = dict(zip(range(19), colors))
                    
        # print('Output shape: ',outputs.shape)
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        decoded = self.decode_segmap(temp=pred,label_colours=label_colours)
        
        return decoded
