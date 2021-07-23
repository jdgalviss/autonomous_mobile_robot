from dl_perception import PerceptionSystem
from costmap import CostMap
from planner import PotentialFieldPlanner
import matplotlib.pyplot as plt
import math
import numpy as np

OSCILLATIONS_DETECTION_LENGTH = 3

class NavigationSystem(object):
    def __init__(self, log = False): 
        self.perception_ = PerceptionSystem()
        self.costmap_ = CostMap(self.perception_.M_, log)
        self.planner_ = PotentialFieldPlanner(self.perception_.M_inv_)

    def transform_path(self, path, robot_state):
        yaw = robot_state[2] - math.pi/2.0
        path_x = path[:,0]*math.cos(yaw) - path[:,1]*math.sin(yaw)
        path_y = path[:,0]*math.sin(yaw) + path[:,1]*math.cos(yaw)
        path = np.array([path_x,path_y])
        path = np.transpose(path)
        path += robot_state[:2]
        print(path.shape)
        return path

    def path_planning(self, img, robot_state):
        driveable_decoded, driveable_mask, preds, driveable_mask_with_objects = self.perception_.process_frame(img)
        cost,cost_obst = self.costmap_.calculate_costmap(driveable_mask, preds, driveable_mask_with_objects)
        path, path_img = self.planner_.calculate_path(cost)
        # print(path)
        # Transform path to global frame
        if robot_state is not None:
            path = self.transform_path(path, robot_state)

        result_img, result_birdview = self.planner_.draw_result(img, cost_obst, path_img, driveable_decoded)
        return path, result_img, result_birdview

#     def seg2scan(self, driveable_area):
#         h,w,_ = driveable_area.shape
#         warped = cv2.warpPerspective(driveable_area, M, (480, 480), flags=cv2.INTER_LINEAR)
#         original_center = np.array([[[w/2,h]]],dtype=np.float32)
#         warped_center = cv2.perspectiveTransform(original_center, M)[0][0]
#         scan_distances, angle_increment, warped_contours = warped2scan(warped, warped_center)
#         return warped, warped_contours, scan_distances, angle_increment