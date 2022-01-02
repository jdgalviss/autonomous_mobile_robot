from dl_perception import PerceptionSystem
from costmap import CostMap
from planner import PotentialFieldPlanner
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import time

class NavigationSystem(object):
    def __init__(self, config): 
        self.perception_ = PerceptionSystem(config)
        self.costmap_ = CostMap(config)
        self.planner_ = PotentialFieldPlanner(config)
        self.config_ = config
        # Local Control Variables
        self.heading_error_int_ = 0.0
        self.prev_heading_error = 0.0

    def transform_path(self, path, robot_state):
        yaw = robot_state[2] - math.pi/2.0
        path_x = path[:,0]*math.cos(yaw) - path[:,1]*math.sin(yaw)
        path_y = path[:,0]*math.sin(yaw) + path[:,1]*math.cos(yaw)
        path = np.array([path_x,path_y])
        path = np.transpose(path)
        path += robot_state[:2]
        return path

    def global_planner_step(self, img, robot_state):
        drivable_img, drivable_edge_points_top, preds, driveable_edge_top_with_objects, segmented_img = self.perception_.process_frame(img)

        ### TIME
        start = time.time()
        cost,cost_obst, cost_fwd, cost_center, lines = self.costmap_.calculate_costmap(drivable_edge_points_top, preds, driveable_edge_top_with_objects, drivable_img)
        end = time.time()
        print("costmap: {} seconds".format(end-start))

        start = time.time()
        path, path_top_img = self.planner_.calculate_path(cost)
        end = time.time()
        print("planner_: {} seconds".format(end-start))
        
        # Transform path to global frame
        if robot_state is not None:
            path = self.transform_path(path, robot_state)

        result_img, result_top = self.planner_.draw_result(img, cost_obst, path_top_img, drivable_img, driveable_edge_top_with_objects)
        
        # if(self.config_.debug):
        #     segmented_img = cv2.addWeighted(img, 0.8, img_decoded, 0.5, 0)  
        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(segmented_img)
        #     plt.show()

        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(cost)
        #     x_idxs = path[:,0]*self.config_.pixel_per_meter_x+self.config_.width/2
        #     y_idxs = -(path[:,1]*self.config_.pixel_per_meter_y-self.config_.height)
        #     ax.plot(x_idxs,y_idxs, color='lightcoral', linewidth=4)
        #     plt.show()

        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(drivable_img)
        #     plt.show()
            
        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(result_img)
        #     plt.show()
            
        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(result_top)
        #     plt.show()
        

        # return path, segmented_img, result_top
        if(self.config_.debug):
            return path, result_img, result_top, segmented_img, drivable_img, cost_fwd, cost_obst, cost_center, cost, path_top_img, lines
        else:
            return path, result_img, result_top, segmented_img


    def local_planner_step(self, robot_state_, global_plan):
        ### TIME
        start = time.time()
        distances = (robot_state_[0]-global_plan[:,0])**2 + (robot_state_[1]-global_plan[:,1])**2
        min_idx = np.argmin(distances) + self.config_.look_ahead
        min_idx = min(min_idx, distances.shape[0]-1)
        local_goal = global_plan[min_idx]

        # Pure pursuit local navigation
        pose_error = local_goal - robot_state_[:2]
        heading_goal = math.atan2(pose_error[1],pose_error[0])
        heading_error = heading_goal - robot_state_[2]

        heading_error = math.atan2(math.sin(heading_error),math.cos(heading_error))

        self.heading_error_int_ += heading_error
        yaw_rate_cmd = self.config_.Kp*(heading_error + self.config_.Ki*self.heading_error_int_ + self.config_.Kd*(heading_error-self.prev_heading_error))
        self.prev_heading_error = heading_error
        if(abs(heading_error)<self.config_.heading_threshold*math.pi/180.0):
            vel_cmd = 1.0*distances[min_idx]
            if (vel_cmd > self.config_.max_speed): vel_cmd = self.config_.max_speed 
        else:
            vel_cmd = 0.0
        # print('vel_cmd: {}'.format(vel_cmd))
        # print('yaw_rate_cmd: {}'.format(yaw_rate_cmd))
        end = time.time()
        print("motion_control: {} seconds".format(end-start))

        return vel_cmd, yaw_rate_cmd
        

#     def seg2scan(self, driveable_area):
#         h,w,_ = driveable_area.shape
#         warped = cv2.warpPerspective(driveable_area, M, (480, 480), flags=cv2.INTER_LINEAR)
#         original_center = np.array([[[w/2,h]]],dtype=np.float32)
#         warped_center = cv2.perspectiveTransform(original_center, M)[0][0]
#         scan_distances, angle_increment, warped_contours = warped2scan(warped, warped_center)
#         return warped, warped_contours, scan_distances, angle_increment