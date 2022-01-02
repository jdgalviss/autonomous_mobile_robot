import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

class CostMap(object):
    def __init__(self, config):
        # Temporary
        mtxs = np.load(config.perspective_transform_path)
        self.M_ = mtxs['M']
        self.h_orig_ = config.original_height
        self.w_orig_ = config.original_width
        self.config_ = config
        
        
    def calculate_costmap(self,drivable_edge_points_top, preds, driveable_edge_top_with_objects, drivable_segmented):
        # Find sidewalk edge lines and angle
        # h,w = driveable_mask.shape
        angle_avg, m_avg, b_avg, lines = self.sidewalk_lines(drivable_edge_points_top, driveable_edge_top_with_objects, drivable_segmented) #driveable_edge_top_with_objects contains all objects and lines
        
        ## Find distance to center cost
        cost_center = self.center_cost(m_avg,b_avg)

        # Create obstacle cost map
        cost_obst = self.obstacle_cost(driveable_edge_top_with_objects, self.config_.obstacle_inflation)

        # Create inclination plane (forward cost)
        cost_forward = self.forward_cost(angle_avg)

        # Total cost
        cost_fcn = cost_obst*self.config_.obstacle_weight+cost_forward*self.config_.forward_weight+cost_center*self.config_.center_weight

        # if(self.config_.debug):
        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(drivable_edge_points_top)
        #     plt.show()
            
        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(driveable_edge_top_with_objects)
        #     plt.show()
            
        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(cost_obst)
        #     plt.show()

        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(cost_center)
        #     plt.show()

        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(cost_forward)
        #     plt.show()

        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     ax.imshow(cost_fcn)
        #     plt.show()

        #     fig = plt.figure(figsize=(14, 7))
        #     ax = plt.axes(projection='3d')
        #     x = np.arange(480)
        #     y = -np.arange(480)
        #     X, Y = np.meshgrid(x, y)
        #     Z = cost_fcn.reshape(X.shape)

        #     ax.plot_surface(X, Y, Z,cmap='viridis', edgecolor='none')
        #     ax.set_title('Surface plot')
        #     ax.view_init(40, -70)
        #     plt.show()

        return cost_fcn,cost_obst, cost_forward, cost_center, lines
                
    def gaus2d(self, x=0, y=0, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
    
    def obstacle_cost(self, mask_with_objects, gaussian_shape = 125):
        x = np.linspace(-2, 2,gaussian_shape)
        y = np.linspace(-2, 2,int(gaussian_shape*2.5))
        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
        z = self.gaus2d(x, y)
        z = np.float32(z)
        cost_obst = cv2.filter2D(mask_with_objects,-1,z)
        cost_obst/=np.amax(cost_obst)
        return cost_obst

    def forward_cost(self, angle):
        normal = np.array([math.tan(-angle),-1,self.config_.height])
        point = np.array([self.config_.width/2, self.config_.height, 0])
        d = -np.sum(point*normal)# dot product
        xx, yy = np.meshgrid(range(self.config_.width), range(self.config_.height))
        cost_forward = (-normal[0]*xx - normal[1]*yy - d)*1./normal[2]
        cost_forward -= np.amin(cost_forward)

        return cost_forward/np.amax(cost_forward)

    def center_cost(self, m,b):
        if m is not None:
            xx, yy = np.meshgrid(range(self.config_.width), range(self.config_.height))
            cost_center = abs(-m*xx+yy-b)/math.sqrt(m**2+1)
        else:
            print("zeros")
            cost_center = np.zeros((self.config_.width,self.config_.height))
            return cost_center
        return cost_center/np.amax(np.abs(cost_center))

    def reject_outliers(self, data, m=2):
        data_filtered = data[abs(data[:,0] - np.mean(data[:,0])) < m * np.std(data[:,0])]
        return data_filtered[abs(data_filtered[:,4] - np.mean(data_filtered[:,4])) < m * np.std(data_filtered[:,4])]

    def sidewalk_lines(self, mask, mask_out, drivable_segmented):
        mask = np.uint8(mask)
        
        # Detect lines in driveable area mask
        lines = cv2.HoughLinesP(mask, 1, 1*np.pi / 180, 100, None, self.config_.min_line_length, 70)
        # lines = cv2.HoughLinesP(mask, 1, 1*np.pi / 180, 100, None, 200, 70)


        line_angles = []
        lines_left = []
        lines_right = []
        lines_found = True
        angles_horizontal = []
        if(lines is not None):
            for line in lines:
                x2,y2,x1,y1 = line[0]
                # Calculate angle, checking which y-coordinate is higher
                if(y2<y1): 
                    angle = -math.atan2(y2-y1,x2-x1)-math.pi/2
                else:
                    x1_aux = x1
                    y1_aux  = y1
                    x1 = x2
                    y1 = y2
                    x2 = x1_aux
                    y2 = y1_aux
                    angle = -math.atan2(y1-y2,x1-x2)-math.pi/2
                angle = math.atan2(math.sin(angle), math.cos(angle))
                line_angles.append(angle)
                    
                # Detect horizontal lines corresponding to the corners
                if((abs(math.atan2(math.sin(angle-math.pi/2), math.cos(angle-math.pi/2))) < self.config_.horizontal_angle_threshold*math.pi/180.0)):
                    # angle = -angle
                    angles_horizontal.append(angle)
                    cv2.line(mask_out,(round(x1),round(y1)),(round(x2),round(y2)),250,1)
                    is_horizontal = True

                elif((abs(math.atan2(math.sin(angle+math.pi/2), math.cos(angle+math.pi/2))) < self.config_.horizontal_angle_threshold*math.pi/180.0)):
                    angles_horizontal.append(angle)
                    cv2.line(mask_out,(round(x1),round(y1)),(round(x2),round(y2)),250,1)
                    is_horizontal = True
                else:
                    is_horizontal = False
                # cv2.line(mask_out,(round(x1),round(y1)),(round(x2),round(y2)),100,1)
                
            
                # Detect coordinate at the bottom of image
                if(x2 == x1): x2 += 1 # Avoid dividing by 0
                if(y2 == y1): y2 += 1 # Avoid dividing by 0
                    
                m = (y2-y1)/(x2-x1)
                b = y1-m*x1
                x1 = round((self.config_.height-b)/m)
                y1 = self.config_.height

                # Detect coordinate at top
                y3 = 0
                x3 = round(-b/m)

                # Add lines to the left and right list
                if(x1<self.config_.width/2 and not is_horizontal): # Condition valid when the robot is going counter clockwise
                    lines_left.append([x1,y1,x2,y2,x3,y3])
                    # lines_left.append([m,b,x2,y2,x1,y1])
                else:
                    lines_right.append([x1,y1,x2,y2,x3,y3])
                    # lines_right.append([m,b,x2,y2,x1,y1])

            lines_left = np.array(lines_left)
            lines_right = np.array(lines_right)
            # print('lines_left: {}'.format(np.array(lines_left)))
            # print('lines_right: {}'.format(np.array(lines_right)))

            if(lines_left.shape[0]>0): lines_left = self.reject_outliers(lines_left)
            if(lines_right.shape[0]>0): lines_right = self.reject_outliers(lines_right)

            # print('lines_left: {}'.format(np.array(lines_left)))
            # print('lines_right: {}'.format(np.array(lines_right)))

            # Find average line left and right
            if(len(lines_left)>0):
                # m_avg_left = np.average(lines_left[:,0])
                x1_left = np.average(lines_left[:,0])
                y1_left = self.config_.height
                x3 = np.average(lines_left[:,4])
                y3 = 0
                if(x3 == x1_left): x3+=1
                m_avg_left = (y3-y1_left)/(x3-x1_left)
                # print('si')
                # print(m_avg_left)
                # print(x3)
                # print(x1_left)


                b_avg_left = y1_left-m_avg_left*x1_left
                y2_left = np.amin(lines_left[:,3])
                x2_left = (y2_left-b_avg_left) / m_avg_left
                if (len(lines_right) > 0): # both lines are found

                    x1_right = np.average(lines_right[:,0])
                    y1_right = self.config_.height
                    x3 = np.average(lines_right[:,4])
                    y3 = 0
                    if(x3 == x1_right): x3+=1

                    m_avg_right = (y3-y1_right)/(x3-x1_right)
                    b_avg_right = y1_right-m_avg_right*x1_right
                    y2_right = np.amin(lines_right[:,3])
                    x2_right = (y2_right-b_avg_right) / m_avg_right
                else:
                    print('right line not found, adding it')
                    m_avg_right = m_avg_left  
                    b_avg_right = -self.config_.avg_driveable_area_width*math.sqrt(m_avg_right**2+1)+b_avg_left
                    if((m_avg_right) < 0):
                        b_avg_right = self.config_.avg_driveable_area_width*math.sqrt(m_avg_right**2+1)+b_avg_left
                    y1_right = y1_left
                    x1_right = (y1_right - b_avg_right) / m_avg_right
                    y2_right = y2_left
                    x2_right = (y2_right - b_avg_right) / m_avg_right
            else:
                if (len(lines_right) > 0): # only right line found
                    print('left line not found, adding it')
                        
                    x1_right = np.average(lines_right[:,0])
                    y1_right = self.config_.height
                    x3 = np.average(lines_right[:,4])
                    y3 = 0
                    if(x3 == x1_right): x3+=1

                    m_avg_right = (y3-y1_right)/(x3-x1_right)
                    b_avg_right = y1_right-m_avg_right*x1_right
                    y2_right = np.amin(lines_right[:,3])
                    x2_right = (y2_right-b_avg_right) / m_avg_right

                    m_avg_left = m_avg_right  
                    b_avg_left = self.config_.avg_driveable_area_width*math.sqrt(m_avg_left**2+1)+b_avg_right
                    if((m_avg_left) < 0):
        #                 print("si")
                        b_avg_left = -self.config_.avg_driveable_area_width*math.sqrt(m_avg_right**2+1)+b_avg_right
                    y1_left = y1_right
                    x1_left = (y1_left - b_avg_left) / m_avg_left
                    if (x1_left > self.config_.width*0.45): # For horizontal lines mostly (turn left only)
                        x1_left = self.config_.width*0.45
                        y1_left = m_avg_left*x1_left + b_avg_left

                    y2_left = y2_right
                    x2_left = (y2_left - b_avg_left) / m_avg_left
                else:
                    angle_avg = 0
                    lines_found = False
        else:
            lines_found = False
        if(len(line_angles) > 0):
            angle_avg = np.average(line_angles)
        else:
            angle_avg = 0.0

        if(lines_found):
            lines = [[x1_right, y1_right, x2_right, y2_right], [x1_left, y1_left, x2_left, y2_left]]
            # print([x1_right, y1_right, x2_right, y2_right])
            # print([x1_left, y1_left, x2_left, y2_left])
            angle_left = -math.atan2(y1_left-y2_left,x1_left-x2_left)-math.pi/2
            angle_right = -math.atan2(y1_right-y2_right,x1_right-x2_right)-math.pi/2

            angle_left = math.atan2(math.sin(angle_left), math.cos(angle_left))
            angle_right = math.atan2(math.sin(angle_right), math.cos(angle_right))



            cv2.line(mask_out,(round(x1_left),round(y1_left)),(round(x2_left),round(y2_left)),250,8)
            cv2.line(mask_out,(round(x1_right),round(y1_right)),(round(x2_right),round(y2_right)),250,8)
            # Calculate middle line
            if len(angles_horizontal) > 0:
                # angle_horizontal = np.average(np.array(angles_horizontal))
                # angle_avg = (angle_left + angle_right + 4.0*angle_horizontal) / 6.0
                angle_avg = (angle_left + angle_right) / 2.0

                if angle_avg < 0.0:
                    angle_avg += math.pi
                # print(angle_horizontal)

            else:
                angle_avg = (angle_left + angle_right) / 2.0
            angle_avg = math.atan2(math.sin(angle_avg), math.cos(angle_avg))
            if((angle_avg>math.pi/2 and angle_avg<math.pi) or (angle_avg>-math.pi and angle_avg<-math.pi/2)):
                angle_avg += math.pi
            angle_avg = math.atan2(math.sin(angle_avg), math.cos(angle_avg))
            

            angle_avg = max(min(angle_avg, math.pi/4.0), -math.pi/4.0)
            m_avg = math.tan(-angle_avg+math.pi/2.0)
            x_center = ( x1_left + x1_right ) / 2.0
            b_avg = self.config_.height - m_avg*x_center

            # if(self.config_.debug):
            #     # drivable_mask = cv2.merge([drivable_mask*0,drivable_mask, drivable_mask*0])
            #     drivable_mask = cv2.warpPerspective(drivable_segmented, self.M_, (480, 480), flags=cv2.INTER_LINEAR)
            #     cv2.line(drivable_mask,(round(x1_left),round(y1_left)),(round(x2_left),round(y2_left)),(0,0,255),8)
            #     cv2.line(drivable_mask,(round(x1_right),round(y1_right)),(round(x2_right),round(y2_right)),(0,0,255),8)
            #     fig, ax = plt.subplots(figsize=(20, 10))
            #     ax.imshow(drivable_mask)
            #     plt.show()

        else:
            m_avg = None
            b_avg = None
            lines = None
        return angle_avg, m_avg, b_avg, lines