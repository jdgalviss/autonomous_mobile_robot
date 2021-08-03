from dl_perception import PerceptionSystem
from costmap import CostMap
from planner import PotentialFieldPlanner
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2

OSCILLATIONS_DETECTION_LENGTH = 3
LOOK_AHEAD = 45
Kp = 0.35
Ki = 0.001
Kd = 0.0
HEADING_THRESHOLD = 35*math.pi/180.0
MAX_SPEED = 0.45

class NavigationSystem(object):
    def __init__(self, debug = False): 
        self.perception_ = PerceptionSystem()
        self.costmap_ = CostMap(self.perception_.M_, debug)
        self.planner_ = PotentialFieldPlanner(self.perception_.M_inv_)
        # Local Control Variables
        self.heading_error_int_ = 0.0
        self.prev_heading_error = 0.0
        self.debug_ = debug

    def transform_path(self, path, robot_state):
        yaw = robot_state[2] - math.pi/2.0
        path_x = path[:,0]*math.cos(yaw) - path[:,1]*math.sin(yaw)
        path_y = path[:,0]*math.sin(yaw) + path[:,1]*math.cos(yaw)
        path = np.array([path_x,path_y])
        path = np.transpose(path)
        path += robot_state[:2]
        return path

    def global_planner_step(self, img, robot_state):
        driveable_decoded, driveable_mask, preds, driveable_mask_with_objects, img_decoded = self.perception_.process_frame(img)
        cost,cost_obst, driveable_mask = self.costmap_.calculate_costmap(driveable_mask, preds, driveable_mask_with_objects)
        path, path_img = self.planner_.calculate_path(cost)

        # Transform path to global frame
        if robot_state is not None:
            path = self.transform_path(path, robot_state)

        result_img, result_birdview = self.planner_.draw_result(img, cost_obst, path_img, driveable_decoded, driveable_mask_with_objects)
        # return path, driveable_mask, result_birdview
        
        if(self.debug_):
            segmented_img = cv2.addWeighted(img, 0.8, img_decoded, 0.5, 0)  
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(segmented_img)
            plt.show()

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(driveable_decoded)
            plt.show()
            
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(result_img)
            plt.show()
            
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(result_birdview)
            plt.show()

        return path, result_img, result_birdview

    def local_planner_step(self, robot_state_, global_plan):
        distances = (robot_state_[0]-global_plan[:,0])**2 + (robot_state_[1]-global_plan[:,1])**2
        min_idx = np.argmin(distances) + LOOK_AHEAD
        min_idx = min(min_idx, distances.shape[0]-1)
        local_goal = global_plan[min_idx]

        # Pure pursuit local navigation
        pose_error = local_goal - robot_state_[:2]
        heading_goal = math.atan2(pose_error[1],pose_error[0])
        heading_error = heading_goal - robot_state_[2]

        heading_error = math.atan2(math.sin(heading_error),math.cos(heading_error))

        self.heading_error_int_ += heading_error
        yaw_rate_cmd = Kp*(heading_error + Ki*self.heading_error_int_ + Kd*(heading_error-self.prev_heading_error))
        self.prev_heading_error = heading_error
        if(abs(heading_error)<HEADING_THRESHOLD):
            vel_cmd = 1.0*distances[min_idx]
            if (vel_cmd > MAX_SPEED): vel_cmd = MAX_SPEED 
        else:
            vel_cmd = 0.0
        # print('vel_cmd: {}'.format(vel_cmd))
        # print('yaw_rate_cmd: {}'.format(yaw_rate_cmd))
        return vel_cmd, yaw_rate_cmd
        

#     def seg2scan(self, driveable_area):
#         h,w,_ = driveable_area.shape
#         warped = cv2.warpPerspective(driveable_area, M, (480, 480), flags=cv2.INTER_LINEAR)
#         original_center = np.array([[[w/2,h]]],dtype=np.float32)
#         warped_center = cv2.perspectiveTransform(original_center, M)[0][0]
#         scan_distances, angle_increment, warped_contours = warped2scan(warped, warped_center)
#         return warped, warped_contours, scan_distances, angle_increment