import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

HORIZ_ANGLE_THRESHOLD = 20*math.pi/180.0
FORWARD_WEIGHT = 0.25#1.5
CENTER_WEIGHT = 0.5#0.3
OBSTACLE_WEIGHT = 4.0#2.0

HEIGHT=480
WIDTH=480
PIXEL_PER_METER_X = (WIDTH - 2*150)/3.0 #Horizontal distance between src points in the real world ( I assumed 3.0 meters)
PIXEL_PER_METER_Y = (HEIGHT - 30-60)/8.0 #Vertical distance between src points in the real world ( I assumed 6.0 meters)
AVG_SIDEWALK_WIDTH = round(3.9*PIXEL_PER_METER_X)


class CostMap(object):
    def __init__(self, M, debug = False):
        # Temporary
        self.M_ = M
        self.h_orig_ = 720
        self.w_orig_ = 1080
        self.debug_ = debug
        
    def calculate_costmap(self,driveable_mask, preds, driveable_mask_with_objects):
        # Find sidewalk edge lines and angle
        # h,w = driveable_mask.shape
        angle_avg, m_avg, b_avg = self.sidewalk_lines(driveable_mask, driveable_mask_with_objects) #driveable_mask_with_objects contains all objects and lines
        
        ## Find distance to center cost
        cost_center = self.center_cost(m_avg,b_avg)

        # Create obstacle cost map
        cost_obst = self.obstacle_cost(driveable_mask_with_objects)

        # Create inclination plane (forward cost)
        cost_forward = self.forward_cost(angle_avg)

        # Total cost
        cost_fcn = cost_obst*OBSTACLE_WEIGHT+cost_forward*FORWARD_WEIGHT+cost_center*CENTER_WEIGHT

        if(self.debug_):
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(driveable_mask)
            plt.show()
            
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(driveable_mask_with_objects)
            plt.show()
            
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(cost_obst)
            plt.show()

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(cost_center)
            plt.show()

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(cost_forward)
            plt.show()

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(cost_fcn)
            plt.show()

            fig = plt.figure(figsize=(14, 7))
            ax = plt.axes(projection='3d')
            x = np.arange(480)
            y = -np.arange(480)
            X, Y = np.meshgrid(x, y)
            Z = cost_fcn.reshape(X.shape)

            ax.plot_surface(X, Y, Z,cmap='viridis', edgecolor='none')
            ax.set_title('Surface plot')
            ax.view_init(40, -70)
            plt.show()

        display_cost = (cost_obst*OBSTACLE_WEIGHT + CENTER_WEIGHT*cost_center)
        display_cost = display_cost / np.amax(display_cost)
        return cost_fcn, display_cost, driveable_mask_with_objects
        
        # return cost_fcn, cost_obst, driveable_mask_with_objects
        
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
        normal = np.array([math.tan(-angle),-1,HEIGHT])
        point = np.array([WIDTH/2, HEIGHT, 0])
        d = -np.sum(point*normal)# dot product
        xx, yy = np.meshgrid(range(WIDTH), range(HEIGHT))
        cost_forward = (-normal[0]*xx - normal[1]*yy - d)*1./normal[2]
        return cost_forward/np.amax(np.abs(cost_forward))

    def center_cost(self, m,b):
        if m is not None:
            xx, yy = np.meshgrid(range(WIDTH), range(HEIGHT))
            cost_center = abs(-m*xx+yy-b)/math.sqrt(m**2+1)
        else:
            print("zeros")
            cost_center = np.zeros((WIDTH,HEIGHT))
            return cost_center
        return cost_center/np.amax(np.abs(cost_center))

    def reject_outliers(self, data, m=2):
        data_filtered = data[abs(data[:,0] - np.mean(data[:,0])) < m * np.std(data[:,0])]
        return data_filtered[abs(data_filtered[:,4] - np.mean(data_filtered[:,4])) < m * np.std(data_filtered[:,4])]

    def sidewalk_lines(self, mask, mask_out):
        mask = np.uint8(mask)
        
        # Detect lines in driveable area mask
        lines = cv2.HoughLinesP(mask, 1, 1*np.pi / 180, 100, None, 200, 70)

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
                if((abs(math.atan2(math.sin(angle-math.pi/2), math.cos(angle-math.pi/2))) < HORIZ_ANGLE_THRESHOLD)):
                    # angle = -angle
                    angles_horizontal.append(angle)
                    cv2.line(mask_out,(round(x1),round(y1)),(round(x2),round(y2)),250,1)
                    is_horizontal = True

                elif((abs(math.atan2(math.sin(angle+math.pi/2), math.cos(angle+math.pi/2))) < HORIZ_ANGLE_THRESHOLD)):
                    angles_horizontal.append(angle)
                    cv2.line(mask_out,(round(x1),round(y1)),(round(x2),round(y2)),250,1)
                    is_horizontal = True
                # cv2.line(mask_out,(round(x1),round(y1)),(round(x2),round(y2)),100,1)
                else:
                    is_horizontal = False
                
            
                # Detect coordinate at the bottom of image
                if(x2 == x1): x2 += 1 # Avoid dividing by 0
                if(y2 == y1): y2 += 1 # Avoid dividing by 0
                    
                m = (y2-y1)/(x2-x1)
                b = y1-m*x1
                x1 = round((HEIGHT-b)/m)
                y1 = HEIGHT

                # Detect coordinate at top
                y3 = 0
                x3 = round(-b/m)

                # Add lines to the left and right list
                if(x1<WIDTH/2 and not is_horizontal): # Condition valid when the robot is going counter clockwise
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
                y1_left = HEIGHT
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
                    y1_right = HEIGHT
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
                    b_avg_right = AVG_SIDEWALK_WIDTH*math.sqrt(m_avg_right**2+1)+b_avg_left
                    if((b_avg_right - b_avg_left) < 0):
                        b_avg_right = -AVG_SIDEWALK_WIDTH*math.sqrt(m_avg_right**2+1)+b_avg_left
                    y1_right = y1_left
                    x1_right = (y1_right - b_avg_right) / m_avg_right
                    y2_right = y2_left
                    x2_right = (y2_right - b_avg_right) / m_avg_right
            else:
                if (len(lines_right) > 0): # only right line found
                    print('left line not found, adding it')
                        
                    x1_right = np.average(lines_right[:,0])
                    y1_right = HEIGHT
                    x3 = np.average(lines_right[:,4])
                    y3 = 0
                    if(x3 == x1_right): x3+=1

                    m_avg_right = (y3-y1_right)/(x3-x1_right)
                    b_avg_right = y1_right-m_avg_right*x1_right
                    y2_right = np.amin(lines_right[:,3])
                    x2_right = (y2_right-b_avg_right) / m_avg_right

                    m_avg_left = m_avg_right  
                    b_avg_left = -AVG_SIDEWALK_WIDTH*math.sqrt(m_avg_left**2+1)+b_avg_right
                    if((b_avg_left - b_avg_right) < 0):
        #                 print("si")
                        b_avg_left = AVG_SIDEWALK_WIDTH*math.sqrt(m_avg_right**2+1)+b_avg_right
                    y1_left = y1_right
                    x1_left = (y1_left - b_avg_left) / m_avg_left
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
            # print([x1_right, y1_right, x2_right, y2_right])
            # print([x1_left, y1_left, x2_left, y2_left])
            angle_left = -math.atan2(y1_left-y2_left,x1_left-x2_left)-math.pi/2
            angle_right = -math.atan2(y1_right-y2_right,x1_right-x2_right)-math.pi/2

            angle_left = math.atan2(math.sin(angle_left), math.cos(angle_left))
            angle_right = math.atan2(math.sin(angle_right), math.cos(angle_right))



            cv2.line(mask_out,(round(x1_left),round(y1_left)),(round(x2_left),round(y2_left)),180,8)
            cv2.line(mask_out,(round(x1_right),round(y1_right)),(round(x2_right),round(y2_right)),180,8)
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
            b_avg = HEIGHT - m_avg*x_center

        else:
            m_avg = None
            b_avg = None
        return angle_avg, m_avg, b_avg