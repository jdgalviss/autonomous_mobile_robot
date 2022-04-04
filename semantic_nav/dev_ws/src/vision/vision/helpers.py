from collections import OrderedDict
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os

HEIGHT = 480
WIDTH = 480
PIXEL_PER_METER_X = (WIDTH - 2*150)/3.0 #Horizontal distance between src points in the real world ( I assumed 2.7 meters)
PIXEL_PER_METER_Y = (HEIGHT - 30-60)/8.0 #Vertical distance between src points in the real world ( I assumed 8 meters)
angle_range = [-40.0, 40.0]
angle_increment = 1.0
max_distance = 7.0


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def takeSecond(elem):
    return elem[1]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def warp_driveable(warped_img, warped_center):
    warped_center[1] = warped_center[1] + 0.2*PIXEL_PER_METER_Y # Assume view deadzone (value to be found with propper extrinsic calibration)
    lower_limit = np.array([50,50,50])
    upper_limit = np.array([200, 200, 200])
    mask = cv2.inRange(np.uint8(warped_img), lower_limit, upper_limit)
    return mask

def warped2scan(warped_img, warped_center):
    warped_center[1] = warped_center[1] + 0.2*PIXEL_PER_METER_Y # Assume view deadzone (value to be found with propper extrinsic calibration)
    lower_limit = np.array([50,50,50])
    upper_limit = np.array([200, 200, 200])
    mask = cv2.inRange(np.uint8(warped_img), lower_limit, upper_limit)
	
    contours, hierarchy = cv2.findContours(mask*200,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_warped = warped_img.copy()
    #for contour in contours:
    #    cv2.drawContours(warped_img, contour, -1, (255, 0, 0), 6)
    
    # Detect distances and angles to points in contour
    scan_distances = []
    scan_angles = []
    for contour in contours:
        for point in contour:
            distance = math.sqrt(((point[0][0]-warped_center[0])/PIXEL_PER_METER_X)**2 + ((point[0][1]-warped_center[1])/PIXEL_PER_METER_Y)**2)
            angle = -math.atan2((point[0][0] - warped_center[0])/PIXEL_PER_METER_X, (warped_center[1]-point[0][1])/PIXEL_PER_METER_Y)
            if(angle<angle_range[1]*math.pi/180.0 and angle>angle_range[0]*math.pi/180.0):
                cv2.circle(contours_warped, (point[0][0], point[0][1]),int(1.0+distance*3.0),(255,0,0), -1)
            scan_distances.append(distance)
            scan_angles.append(angle)
    
    #arrange contour data
    scan_array = np.array(([scan_distances, scan_angles])).T
    scan_list = list(scan_array)
    scan_list.sort(key=takeSecond)
    scan_array = np.array(scan_list)
    #Resample data to form scan_data
    scan_distances = []
    scan_angles = []
    if scan_array.shape[0]>20:
        for angle in np.arange(angle_range[0], angle_range[1], angle_increment):
            rads = angle*math.pi/180.0
            scan_angles.append(rads)
            idx = find_nearest_idx(scan_array[:,1], rads)
            if(scan_array[idx,0] < max_distance):
                scan_distances.append(scan_array[idx,0])
            else:
                scan_distances.append(0.0)
    return scan_distances, angle_increment*math.pi/180.0, contours_warped
    
def get_driveable_mask(warped, warped_center):
    warped_center[1] = warped_center[1] + 0.2*PIXEL_PER_METER_Y # Assume view deadzone (value to be found with propper extrinsic calibration)
    # Calculate driveable area mask
    lower_limit = np.array([50,50,50])
    upper_limit = np.array([200, 200, 200])
    driveable_mask = cv2.inRange(np.uint8(warped), lower_limit, upper_limit)

    # Find contours corresponding to driveable area limits
    contours, hierarchy = cv2.findContours(driveable_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_mask = np.zeros(driveable_mask.shape)
    contours_mask2 = np.zeros(driveable_mask.shape)
    
    #for contour in contours:
    cv2.drawContours(contours_mask2, contours, -1, (255, 255, 255), 3)

    for contour in contours:
        for point in contour:
            distance = math.sqrt(((point[0][0]-warped_center[0])/PIXEL_PER_METER_X)**2 + ((point[0][1]-warped_center[1])/PIXEL_PER_METER_Y)**2)
            angle = -math.atan2((point[0][0] - warped_center[0])/PIXEL_PER_METER_X, (warped_center[1]-point[0][1])/PIXEL_PER_METER_Y)
            if(angle<angle_range[1]*math.pi/180.0 and angle>angle_range[0]*math.pi/180.0):
                cv2.circle(contours_mask, (point[0][0], point[0][1]),20,15, -1)
                scan_distances.append(distance)
                scan_angles.append(angle)
            else:
                cv2.circle(contours_mask2, (point[0][0], point[0][1]),6,0, -1)
    contours_mask2[:3,:] = 0
    contours_mask2[-2:,:] = 0
    
#                 contours_mask[point[0][1], point[0][0]] = 255
    return contours_mask2

def get_driveable_mask2(warped, warped_center, config):
    angle_range = [-config.angle_range, config.angle_range]

    warped_center[1] = warped_center[1] + 0.1*config.pixel_per_meter_y # Assume view deadzone (value to be found with propper extrinsic calibration)
    # Calculate driveable area mask
    lower_limit = np.array([50,50,50])
    upper_limit = np.array([200, 200, 200])
    driveable_mask = cv2.inRange(np.uint8(warped), lower_limit, upper_limit)

    # Find contours corresponding to driveable area limits
    contours, hierarchy = cv2.findContours(driveable_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_mask = np.zeros(driveable_mask.shape)

    # Detect distances and angles to points in contour
    scan_points = []
    scan_distances = []
    scan_angles = []
    for contour in contours:
        for point in contour:
            distance = math.sqrt(((point[0][0]-warped_center[0])/config.pixel_per_meter_x)**2 + ((point[0][1]-warped_center[1])/config.pixel_per_meter_y)**2)
            angle = -math.atan2((point[0][0] - warped_center[0])/config.pixel_per_meter_x, (warped_center[1]-point[0][1])/config.pixel_per_meter_y)
            if(angle<angle_range[1]*math.pi/180.0 and angle>angle_range[0]*math.pi/180.0):
                scan_points.append(np.array([point[0][0], point[0][1]]))
                scan_distances.append(distance)
                scan_angles.append(angle)
           
    points_definitive = []   
    scan_array = np.array(([scan_distances, scan_angles, scan_points])).T
    
    scan_list = list(scan_array)
    scan_list.sort(key=takeSecond)
    scan_array = np.array(scan_list)
    prev_angle = 10000
    angle_thresh = 0.1*math.pi/180.0
    min_d = 1000
    for scan in scan_array:
        d = scan[0]
        angle = scan[1]
        point = scan[2]
         
        if(abs(angle-prev_angle)<angle_thresh):
            if(d<min_d):
                points_definitive[-1]=point
                min_d = d
        else:
            points_definitive.append(point)
            min_d = d
        prev_angle = angle
    for p in points_definitive:
        if(p[0] != int(config.width/2) and (p[1]!= config.height)):
            # cv2.circle(contours_mask, (p[0], p[1]),8,255, -1)
            cv2.circle(contours_mask, (p[0], p[1]),config.edge_point_size,255, -1)
        
    return contours_mask


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        #print(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
            
    return images, filenames

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw