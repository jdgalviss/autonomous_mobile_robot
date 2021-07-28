import cv2
from collections import deque
import numpy as np
import math

OSCILLATIONS_DETECTION_LENGTH = 3
HEIGHT=480
WIDTH=480
PIXEL_PER_METER_X = (WIDTH - 2*150)/3.0 #Horizontal distance between src points in the real world ( I assumed 3.0 meters)
PIXEL_PER_METER_Y = (HEIGHT - 30-60)/8.0 #Vertical distance between src points in the real world ( I assumed 6.0 meters)

class PotentialFieldPlanner(object):
                          
    def __init__(self, M_inv):
        self.M_inv_ = M_inv
        self.motion_model = self.get_motion_model(5)
        print("Planner")
                         
    def get_motion_model(self, size):
        # dx, dy
        motion_model = []
        for i in range(-size,size):
            for j in range(1,size):
                if(i != 0):
                    motion_model.append([i,-j])

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
        if (len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH):
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
        ix = round(WIDTH/2)
        iy = round(HEIGHT-5) 
        path = []
        while(ix > 5 and iy > 50 and ix< WIDTH-5):
            minp = float("inf")
            minix, miniy = 0, -1
            for i, _ in enumerate(self.motion_model):
                inx = int(ix + 1*self.motion_model[i][0])
                iny = int(iy + 1*self.motion_model[i][1])
                if inx >= WIDTH or iny >= HEIGHT or inx < 0 or iny < 0:
                    p = float("inf")  # outside area
                    print("outside potential!")
                else:
                    p = pmap[iny][inx] * (math.sqrt(self.motion_model[i][0]**2 + self.motion_model[i][1]**2))

                if minp >= p:
                    minp = p
                    minix = inx
                    miniy = iny
            ix = minix
            iy = miniy
            # Calculate points
            px = (ix-WIDTH/2)/PIXEL_PER_METER_X
            py = (HEIGHT-iy)/PIXEL_PER_METER_Y
            path.append([px,py])
            if (self.oscillations_detection(previous_ids, ix, iy)):
                print("Oscillation detected at ({},{})!".format(ix, iy))
                break
            cv2.circle(output, (ix, iy),int(3.0),1, -1)
        return np.array(path),output

    def draw_result(self, img, cost_obst, path_img, driveable_mask):
        h_orig,w_orig,_ = img.shape

        result_birdview = cv2.merge([cost_obst, path_img, cost_obst*0])
        result_birdview = np.uint8(result_birdview*255.0)
        unwarped_birdview = cv2.warpPerspective(result_birdview, self.M_inv_, (w_orig,h_orig), flags=cv2.INTER_LINEAR)
    #     unwarped_birdview = np.uint8(unwarped_birdview*255.0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = cv2.addWeighted(img, 0.8, unwarped_birdview, 0.5, 0)  
        output = cv2.addWeighted(output, 0.9, driveable_mask, 0.1, 0)    

        return output, result_birdview
