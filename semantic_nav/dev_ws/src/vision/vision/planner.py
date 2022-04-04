import cv2
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt

class PotentialFieldPlanner(object):
                          
    def __init__(self, config):
        self.config_ = config
        mtxs = np.load(config.perspective_transform_path)
        self.M_inv_ = mtxs['M_inv']
        self.motion_model = self.get_motion_model(3,3)
        print("Planner")
                         
    def get_motion_model(self, min,max):
        # dx, dy
        motion_model = []
        for i in range(-max,max+1):
            for j in range(-max,min+1):
                if((j == -max or j==min) or (i == -max) or (i == max)):
                    if(j!=0 and i!= 0):
                        motion_model.append([i,-j])
        # motion_model = []
        # for i in range(-max,max):
        #     for j in range(-max,max):
        #         if(i != 0 and j != 0):
        #             motion_model.append([i,-j])

    #     self.motion_model = [
    #             #   [2, 0], #right
    # #               [0, 2], #back
    #             #   [-2, 0], #left
    #               #[0, -3], # front
    #               [-1, -2],#front-left
    #               [-2, -2],#front-left
    #               [-2, -1],#front-left

    #               [-1, -4],#front-left
    #               [-3, -4],#front-left

    #               [-3, -1],#front-left
    #               [-3, -2],#front-left



    #                 #   [-3, -4],#front-left
    #                 #   [-4, -4],#front-left
    # #               [-1, 2], # back-left
    #               [1, -2], #front-right
    #               [2, -2], #front-right
    #               [2, -1], #front-right

    #               [1, -4],#front-right
    #               [3, -4],#front-right

    #               [3, -1],#front-right
    #               [3, -2],#front-right

    #             #   [3, -4], #front-right
    #             #   [4, -4], #front-right
    # # #               [1, 1]  #back-right
                #   ]
        return motion_model

    def oscillations_detection(self,previous_ids, ix, iy):
        previous_ids.append((ix, iy))
        if (len(previous_ids) > self.config_.oscillations_detection_length):
            previous_ids.popleft()
        # check if contains any duplicates by copying into a set
        previous_ids_set = set()
        for index in previous_ids:
            if index in previous_ids_set:
                return True
            else:
                previous_ids_set.add(index)
        return False

    def calculate_path(self,pmap):
        output = pmap.copy()*0
        # self.motion_model = self.get_motion_model()
        previous_ids = deque()
        ix = round(self.config_.width/2)
        iy = round(self.config_.height-self.config_.start_idx) 
        path = []
        counter = 0
        while(ix > 5 and iy > 50 and ix< self.config_.width-5):
            minp = float("inf")
            minix, miniy = 0, -1
            for i, _ in enumerate(self.motion_model):
                inx = int(ix + 1*self.motion_model[i][0])
                iny = int(iy + 1*self.motion_model[i][1])
                if inx >= self.config_.width or iny >= self.config_.height or inx < 0 or iny < 0:
                    p = float("inf")  # outside area
                    print("outside potential!")
                    break
                else:
                    p = pmap[iny][inx] * (math.sqrt(self.motion_model[i][0]**2 + self.motion_model[i][1]**2))

                if minp >= p:
                    minp = p
                    minix = inx
                    miniy = iny
            ix = minix
            iy = miniy
            # Calculate points
            px = (ix-self.config_.width/2)/self.config_.pixel_per_meter_x
            py = (self.config_.height-iy)/self.config_.pixel_per_meter_y
            path.append([px,py])
            if (self.oscillations_detection(previous_ids, ix, iy)):
                print("Oscillation detected at ({},{})!".format(ix, iy))
                break
            cv2.circle(output, (ix, iy),3,1, -1)

            if counter == self.config_.look_ahead:
                cv2.circle(output, (ix, iy),10,1, -1)
                cv2.putText(output, 'LookAhead', (ix+15, iy+0), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 1, 2, cv2.LINE_AA)
            counter+=1
        if(counter < self.config_.look_ahead):
            cv2.circle(output, (ix, iy),10,1, -1)
            cv2.putText(output, 'LookAhead', (ix+15, iy+0), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 1, 2, cv2.LINE_AA)

        return np.array(path),output

    def draw_result(self, img, cost_obst, path_img, driveable_mask, driveable_mask_with_objects):
        h_orig,w_orig,_ = img.shape
        people_img = np.uint8(220*(driveable_mask_with_objects==253))
        lines_img = np.uint8(240*(driveable_mask_with_objects==250))
        lines_with_people = cv2.merge([people_img*0, people_img*0, lines_img+0*people_img])
        lines_with_people = np.uint8(lines_with_people)
        
        result_birdview = cv2.merge([cost_obst, path_img, cost_obst*0])
        result_birdview = np.uint8(result_birdview*240.0)
        result_birdview = cv2.addWeighted(result_birdview, 0.7, lines_with_people, 1.0, 0) 
        unwarped_birdview = cv2.warpPerspective(result_birdview, self.M_inv_, (w_orig,h_orig), flags=cv2.INTER_LINEAR)
    #     unwarped_birdview = np.uint8(unwarped_birdview*255.0)
        # if(self.config_.model_name == 'fchardnet'):
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = cv2.addWeighted(img, 0.8, unwarped_birdview, 0.5, 0)  
        output = cv2.addWeighted(output, 0.9, driveable_mask, 0.1, 0)    

        return output, result_birdview
