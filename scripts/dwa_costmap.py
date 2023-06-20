#!/usr/bin/env python3

# Author: Connor McGuile
# Latest author: Adarsh Jagan Sathyamoorthy
# Feel free to use in any way.

# A custom Dynamic Window Approach implementation for use with Turtlebot.
# Obstacles are registered by a front-mounted laser and stored in a set.
# If, for testing purposes or otherwise, you do not want the laser to be used,
# disable the laserscan subscriber and create your own obstacle set in main(),
# before beginning the loop. If you do not want obstacles, create an empty set.
# Implentation based off Fox et al.'s paper, The Dynamic Window Approach to
# Collision Avoidance (1997).

import rospy
import math
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import sensor_msgs.msg
from sensor_msgs.msg import LaserScan, CompressedImage
from tf.transformations import euler_from_quaternion
import time
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
import sys
import csv

# Headers for local costmap subscriber
from matplotlib import pyplot as plt
from matplotlib.path import Path
from PIL import Image

import sys
# OpenCV
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


class Config():
    # simulation parameters

    def __init__(self):
        
        # Robot parameters
        self.max_speed = 0.6     # [m/s]
        self.min_speed = 0.0     # [m/s]
        self.max_yawrate = 0.6   # [rad/s]
        self.max_accel = 1       # [m/ss]
        self.max_dyawrate = 3.2  # [rad/ss]
        
        self.v_reso = 0.30 #0.20              # [m/s]
        self.yawrate_reso = 0.20  # [rad/s]
        
        self.dt = 0.5  # [s]
        self.predict_time = 2.0 #3.0 #1.5  # [s]
        
        # 1===
        self.to_goal_cost_gain = 5.0       # lower = detour
        self.veg_cost_gain = 1.0
        self.speed_cost_gain = 0.1   # 0.1   # lower = faster
        self.obs_cost_gain = 3.2            # lower z= fearless
        
        self.robot_radius = 0.6  # [m]
        self.x = 0.0
        self.y = 0.0
        self.v_x = 0.0
        self.w_z = 0.0
        self.goalX = 0.0006
        self.goalY = 0.0006
        self.th = 0.0
        self.r = rospy.Rate(20)

        self.collision_threshold = 0.3 # [m]

        # DWA output
        self.min_u = []

        self.stuck_status = False
        self.happend_once = False
        self.stuck_count = 0
        self.pursuing_safe_loc = False
        self.okay_locations = []
        self.stuck_locations = []


        # Costmap
        self.scale_percent = 300 # percent of original size
        self.costmap_shape = (200, 200)
        self.costmap_resolution = 0.05
        print("Initialized Costmap!")
        self.costmap_baselink_high = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.costmap_baselink_mid = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.costmap_baselink_low = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.costmap_rgb = cv2.cvtColor(self.costmap_baselink_low,cv2.COLOR_GRAY2RGB)
        self.obs_low_mid_high = np.argwhere(self.costmap_baselink_low > 150) # should be null set

        # For cost map clearing
        self.height_thresh = 75#150
        self.alpha = 0.35
        
        # For on-field visualization
        self.plan_map_pub = rospy.Publisher("/planning_costmap", sensor_msgs.msg.Image, queue_size=10)
        self.viz_pub = rospy.Publisher("/viz_costmap", sensor_msgs.msg.Image, queue_size=10) 
        self.br = CvBridge()


    # Callback for Odometry
    def assignOdomCoords(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll,pitch,theta) = euler_from_quaternion ([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
        # (roll,pitch,theta) = euler_from_quaternion ([rot_q.z, -rot_q.x, -rot_q.y, rot_q.w]) # used when lego-loam is used
        
        self.th = theta
        # print("Theta of body wrt odom:", self.th/0.0174533)

        # Get robot's current velocities
        self.v_x = msg.twist.twist.linear.x
        self.w_z = msg.twist.twist.angular.z 
        # print("Robot's current velocities", [self.v_x, self.w_z])



    # Callback for goal from POZYX
    def target_callback(self, data):
        print("---------------Inside Goal Callback------------------------")

        radius = data.linear.x # this will be r
        theta = data.linear.y * 0.0174533 # this will be theta
        print("r and theta:",data.linear.x, data.linear.y)
        
        # Goal wrt robot frame        
        goalX_rob = radius * math.cos(theta)
        goalY_rob = radius * math.sin(theta)

        # Goal wrt odom frame (from where robot started)
        self.goalX =  self.x + goalX_rob*math.cos(self.th) - goalY_rob*math.sin(self.th)
        self.goalY = self.y + goalX_rob*math.sin(self.th) + goalY_rob*math.cos(self.th)
        
        # print("Self odom:",self.x, self.y)
        # print("Goals wrt odom frame:", self.goalX, self.goalY)

        # If goal is published as x, y coordinates wrt odom uncomment this
        # self.goalX = data.linear.x
        # self.goalY = data.linear.y



    # Callback for local costmap from move_base and converting it wrt robot frame
    def high_costmap_callback(self, data):

        # print("Received high local costmap!")

        costmap_2d = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        costmap_2d = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        costmap_2d = np.rot90(np.fliplr(costmap_2d), 1, (1, 0))

        cm_image = Image.fromarray(np.uint8(costmap_2d))
        yaw_deg = 0 # Now cost map published wrt baselink
        cm_baselink_pil = cm_image.rotate(-yaw_deg)
        self.costmap_baselink_high = np.array(cm_baselink_pil)


    def mid_costmap_callback(self, data):

        # print("Received mid local costmap!")

        costmap_2d = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        costmap_2d = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        costmap_2d = np.rot90(np.fliplr(costmap_2d), 1, (1, 0))

        cm_image = Image.fromarray(np.uint8(costmap_2d))

        # yaw_deg = self.th*180/math.pi
        yaw_deg = 0
        cm_baselink_pil = cm_image.rotate(-yaw_deg)
        self.costmap_baselink_mid = np.array(cm_baselink_pil)


    def low_costmap_callback(self, data):

        # print("Received low local costmap!")

        costmap_2d = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        costmap_2d = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        costmap_2d = np.rot90(np.fliplr(costmap_2d), 1, (1, 0))

        cm_image = Image.fromarray(np.uint8(costmap_2d))

        # yaw_deg = self.th*180/math.pi
        yaw_deg = 0 # check costmap_local.yaml. The frame has been changed from odom to base_link
        cm_baselink_pil = cm_image.rotate(-yaw_deg)
        self.costmap_baselink_low = np.array(cm_baselink_pil)
        self.costmap_rgb = cv2.cvtColor(self.costmap_baselink_low, cv2.COLOR_GRAY2RGB)

        # Robot location on costmap
        rob_x = int(self.costmap_rgb.shape[0]/2)
        rob_y = int(self.costmap_rgb.shape[1]/2)

        # VISUALIZATION
        # Mark the robot on costmap 
        self.costmap_rgb = cv2.circle(self.costmap_rgb, (rob_x, rob_y), 4, (255, 0, 255), -1)
        self.costmap_sum()
        
        # dim = (int(self.costmap_baselink_low.shape[1] * self.scale_percent / 100), int(self.costmap_baselink_low.shape[0] * self.scale_percent / 100)) 
        # resized = cv2.resize(self.costmap_rgb, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow('costmap_wrt_robot', resized)
        # cv2.waitKey(3)


    def costmap_sum(self):
        # costmap_sum = self.costmap_baselink_low + self.costmap_baselink_mid + self.costmap_baselink_high
        costmap_sum = self.costmap_baselink_high
        self.obs_low_mid_high = np.argwhere(costmap_sum > self.height_thresh) # (returns row, col)

        if(self.obs_low_mid_high.shape[0] != 0):
            self.costmap_rgb = self.tall_obstacle_marker(self.costmap_rgb, self.obs_low_mid_high)
        else:
            pass


    def tall_obstacle_marker(self, rgb_image, centers):
        # Marking centers red = (0, 0, 255), or orange = (0, 150, 255)
        rgb_image[centers[:, 0], centers[:, 1], 0] = 0
        rgb_image[centers[:, 0], centers[:, 1], 1] = 0
        rgb_image[centers[:, 0], centers[:, 1], 2] = 255
        return rgb_image


    # 2===
    def classification_callback(self, data):

        # print("Received classification results!")

        # Define grid cells belonging to each quadrant of the image
        # (col, row) convention
        # top_left = [(84, 0), (100, 49)]
        # top_right = [(100, 0), (117, 49)]
        # bottom_left = [(84, 49), (100, 82)]
        # bottom_right = [(100, 49), (117, 82)]

        # Q1 = np.array([(col, row) for col in range(84, 100+1) for row in range(0, 49+1)])
        # Q2 = np.array([(col, row) for col in range(100, 117+1) for row in range(0, 49+1)])
        # Q3 = np.array([(col, row) for col in range(84, 100+1) for row in range(49, 82+1)])
        # Q4 = np.array([(col, row) for col in range(100, 117+1) for row in range(49, 82+1)])

        # Sanity check for modifying costmap for navigation. THIS IS THE CORRECT CONVENTION.
        # Note: (row, col) convention is used for np array
        # self.costmap_baselink_low[Q1[:,1], Q1[:,0]] = 200
        # self.costmap_baselink_low[Q2[:,1], Q2[:,0]] = 255
        # self.costmap_baselink_low[Q3[:,1], Q3[:,0]] = 255
        # self.costmap_baselink_low[Q4[:,1], Q4[:,0]] = 200
        # cv2.imshow("Modified Costmap", self.costmap_baselink_low)
        # cv2.waitKey(3)

        # Sanity check for modifying costmap for visualization. THIS IS THE CORRECT CONVENTION.
        # cv2.rectangle(self.costmap_rgb, pt1=(84, 0), pt2=(100, 49), color=(0,255,0), thickness= 1) 
        # cv2.rectangle(self.costmap_rgb, pt1=(100, 0), pt2=(117, 49), color=(0,255,0), thickness= 1) 
        # cv2.rectangle(self.costmap_rgb, pt1=(84, 49), pt2=(100, 82), color=(255,0,0), thickness= 1) 
        # cv2.rectangle(self.costmap_rgb, pt1=(100, 49), pt2=(117, 82), color=(255,0,0), thickness= 1) 
        # dim = (int(self.costmap_rgb.shape[1] * self.scale_percent / 100), int(self.costmap_rgb.shape[0] * self.scale_percent / 100)) 
        # resized = cv2.resize(self.costmap_rgb, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow('costmap', resized)
        # cv2.waitKey(3)

        top_left = [(84, 0), (100, 49)]
        top_right = [(100, 0), (117, 49)]
        bottom_left = [(84, 49), (100, 100)]
        bottom_right = [(100, 49), (117, 100)]

        Q1 = np.array([(col, row) for col in range(84, 100+1) for row in range(0, 49+1)])
        Q2 = np.array([(col, row) for col in range(100, 117+1) for row in range(0, 49+1)])
        Q3 = np.array([(col, row) for col in range(84, 100+1) for row in range(49, 100+1)])
        Q4 = np.array([(col, row) for col in range(100, 117+1) for row in range(49, 100+1)])

        # Clear Costmap
        # NOTE: Modify this based on the actual data being published
        # 0 - non-pliable, 1 - pliable
        # Even indices correspond to class number, Odd indices correspond to distance
        veg1 = data.data[0]
        veg2 = data.data[2]
        veg3 = data.data[4]
        veg4 = data.data[6]
        conf1 = math.exp(-self.alpha * data.data[1])
        conf2 = math.exp(-self.alpha * data.data[3])
        conf3 = math.exp(-self.alpha * data.data[5])
        conf4 = math.exp(-self.alpha * data.data[7])
        # print(conf1, conf2, conf3, conf4)

        conf_thresh = 0.90

        if (veg1 == 1 and conf1 >= conf_thresh):
            # self.costmap_baselink_low[Q1[:,1], Q1[:,0]] = (self.costmap_baselink_low[Q1[:,1], Q1[:,0]] * (1-conf1))
            self.costmap_baselink_low[Q1[:,1], Q1[:,0]] = 0
            # self.costmap_rgb[Q1[:,1], Q1[:,0], :] = 0
            cv2.rectangle(self.costmap_rgb, pt1=(84, 0), pt2=(100, 49), color=(0,255,0), thickness= 1)

        else:
            # Don't clear costmap
            cv2.rectangle(self.costmap_rgb, pt1=(84, 0), pt2=(100, 49), color=(0,0,255), thickness= 1)


        # Quadrant 2
        if (veg2 == 1 and conf2 >= conf_thresh):
            # self.costmap_baselink_low[Q2[:,1], Q2[:,0]] = (self.costmap_baselink_low[Q2[:,1], Q2[:,0]] * (1-conf2))
            self.costmap_baselink_low[Q2[:,1], Q2[:,0]] = 0
            # self.costmap_rgb[Q2[:,1], Q2[:,0], :] = 0
            cv2.rectangle(self.costmap_rgb, pt1=(100, 0), pt2=(117, 49), color=(0,255,0), thickness= 1)
        else:
            cv2.rectangle(self.costmap_rgb, pt1=(100, 0), pt2=(117, 49), color=(0,0,255), thickness= 1)


        # Quadrant 3
        if (veg3 == 1 and conf3 >= conf_thresh):
            # Clear cost map
            # self.costmap_baselink_low[Q3[:,1], Q3[:,0]] = (self.costmap_baselink_low[Q3[:,1], Q3[:,0]] * (1-conf3))
            self.costmap_baselink_low[Q3[:,1], Q3[:,0]] = 0
            # self.costmap_rgb[Q3[:,1], Q3[:,0], :] = 0
            cv2.rectangle(self.costmap_rgb, pt1=(84, 49), pt2=(100, 100), color=(0,255,0), thickness= 1)
        else:
            cv2.rectangle(self.costmap_rgb, pt1=(84, 49), pt2=(100, 100), color=(0,0,255), thickness= 1)


        # Quadrant 4
        if (veg4 == 1 and conf4 >= conf_thresh):
            # Clear cost map
            # self.costmap_baselink_low[Q4[:,1], Q4[:,0]] = (self.costmap_baselink_low[Q4[:,1], Q4[:,0]] * (1-conf4))
            self.costmap_baselink_low[Q4[:,1], Q4[:,0]] = 0
            # self.costmap_rgb[Q4[:,1], Q4[:,0], :] = 0
            cv2.rectangle(self.costmap_rgb, pt1=(100, 49), pt2=(117, 100), color=(0,255,0), thickness= 1) 
        else:
            cv2.rectangle(self.costmap_rgb, pt1=(100, 49), pt2=(117, 100), color=(0,0,255), thickness= 1) 

        self.costmap_baselink_low = self.costmap_baselink_low.astype('uint8')

        self.plan_map_pub.publish(self.br.cv2_to_imgmsg(self.costmap_baselink_low, encoding="mono8"))
        self.viz_pub.publish(self.br.cv2_to_imgmsg(self.costmap_rgb, encoding="bgr8"))

        # VISUALIZATION
        # cv2.imshow("Modified Costmap", self.costmap_baselink_low)
        # dim = (int(self.costmap_rgb.shape[1] * self.scale_percent / 100), int(self.costmap_rgb.shape[0] * self.scale_percent / 100)) 
        # resized = cv2.resize(self.costmap_rgb, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow('costmap', resized)
        # cv2.waitKey(3)




class Obstacles():
    def __init__(self):
        # Set of coordinates of obstacles in view
        self.obst = set()
        self.collision_status = False

    # Custom range implementation to loop over LaserScan degrees with
    # a step and include the final degree
    def myRange(self,start,end,step):
        i = start
        while i < end:
            yield i
            i += step
        yield end


    # Callback for LaserScan
    def assignObs(self, msg, config):

        deg = len(msg.ranges)   # Number of degrees - varies in Sim vs real world
        self.obst = set()   # reset the obstacle set to only keep visible objects

        maxAngle = 360
        scanSkip = 1
        anglePerSlot = (float(maxAngle) / deg) * scanSkip
        angleCount = 0
        angleValuePos = 0
        angleValueNeg = 0
        self.collision_status = False
        for angle in self.myRange(0,deg-1,scanSkip):
            distance = msg.ranges[angle]

            if (distance < 0.05) and (not self.collision_status):
                self.collision_status = True
                # print("Collided")
                reached = False
                reset_robot(reached)

            if(angleCount < (deg / (2*scanSkip))):
                # print("In negative angle zone")
                angleValueNeg += (anglePerSlot)
                scanTheta = (angleValueNeg - 180) * math.pi/180.0


            elif(angleCount>(deg / (2*scanSkip))):
                # print("In positive angle zone")
                angleValuePos += anglePerSlot
                scanTheta = angleValuePos * math.pi/180.0
            # only record obstacles that are within 4 metres away

            else:
                scanTheta = 0

            angleCount += 1

            if (distance < 4):
                # angle of obstacle wrt robot
                # angle/2.844 is to normalise the 512 degrees in real world
                # for simulation in Gazebo, use angle/4.0
                # laser from 0 to 180


                objTheta =  scanTheta + config.th

                # round coords to nearest 0.125m
                obsX = round((config.x + (distance * math.cos(abs(objTheta))))*8)/8
                # determine direction of Y coord
                
                if (objTheta < 0):
                    obsY = round((config.y - (distance * math.sin(abs(objTheta))))*8)/8
                else:
                    obsY = round((config.y + (distance * math.sin(abs(objTheta))))*8)/8


                # add coords to set so as to only take unique obstacles
                self.obst.add((obsX,obsY))
                


# Model to determine the expected position of the robot after moving along trajectory
def motion(x, u, dt):
    # motion model
    # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt

    x[3] = u[0]
    x[4] = u[1]

    return x


# Determine the dynamic window from robot configurations
def calc_dynamic_window(x, config):

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    #  [vmin, vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    # print("Dynamic Window: ", dw)
    return dw


# Calculate a trajectory sampled across a prediction time
def calc_trajectory(xinit, v, y, config):

    x = np.array(xinit)
    traj = np.array(x)  # many motion models stored per trajectory
    time = 0
    while time <= config.predict_time:
        # store each motion model along a trajectory
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt # next sample

    # print("Trajectory: ", traj)
    # print("End point:", traj[-1, 0], traj[-1, 1])
    # print("Mid point: ", traj[math.floor(len(traj)/2), 0], traj[math.floor(len(traj)/2), 1])

    return traj


# 3===
# Calculate trajectory, costings, and return velocities to apply to robot
def calc_final_input(x, u, dw, config, ob):

    xinit = x[:]
    min_cost = 10000.0
    config.min_u = u
    config.min_u[0] = 0.0
    
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    orange = (0, 150, 255)

    count = 0
    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1] + config.v_reso/2, config.v_reso):
        for w in np.arange(dw[2], dw[3] + config.yawrate_reso/2, config.yawrate_reso):
            count = count + 1 
            
            traj = calc_trajectory(xinit, v, w, config)

            # calc costs with weighted gains
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, config)
            speed_cost = config.speed_cost_gain * (config.max_speed - traj[-1, 3]) # end v should be as close to max_speed to have low cost
            veg_cost = config.veg_cost_gain * calc_veg_cost(traj, config)
            # ob_cost = config.obs_cost_gain * calc_obstacle_cost(traj, ob, config)

            # final_cost = to_goal_cost + veg_cost
            final_cost = to_goal_cost + veg_cost + speed_cost
            # final_cost = to_goal_cost*(1 + veg_cost)
            
            # print(count, "v,w = %.2f %.2f"% (v, w))
            # print("Goal cost = %.2f"% to_goal_cost, "veg_cost = %.2f"% veg_cost, "final_cost = %.2f"% final_cost)
            # print("Goal cost = %.2f"% to_goal_cost, "speed_cost = %.2f"% speed_cost, "veg_cost = %.2f"% veg_cost, "final_cost = %.2f"% final_cost)

            
            config.costmap_rgb = draw_traj(config, traj, yellow)

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                config.min_u = [v, w]

    # print("Robot's current velocities", [config.v_x, config.w_z])
    # traj = calc_trajectory(xinit, config.v_x, config.w_z, config) # This leads to buggy visualization

    traj = calc_trajectory(xinit, config.min_u[0], config.min_u[1], config)
    to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, config)
    veg_cost_min = config.veg_cost_gain * calc_veg_cost(traj, config)
    print("min_u = %.2f %.2f"% (config.min_u[0], config.min_u[1]), "Goal cost = %.2f"% to_goal_cost, "Veg cost = %.2f"% veg_cost_min, "Min cost = %.2f"% min_cost)
    config.costmap_rgb = draw_traj(config, traj, green)

    # VISUALIZATION
    # dim = (int(config.costmap_rgb.shape[1] * config.scale_percent / 100), \
    #  int(config.costmap_rgb.shape[0] * config.scale_percent / 100)) 
    # resized = cv2.resize(config.costmap_rgb, dim, interpolation = cv2.INTER_AREA)
    
    # # cv2.imshow('costmap_wrt_robot', resized)
    # cv2.imshow('costmap_baselink', config.costmap_baselink)
    # cv2.waitKey(3)
    
    return config.min_u
    


# Calculate obstacle cost inf: collision, 0:free
def calc_obstacle_cost(traj, ob, config):
    skip_n = 2
    minr = float("inf")

    # Loop through every obstacle in set and calc Pythagorean distance
    # Use robot radius to determine if collision
    for ii in range(0, len(traj[:, 1]), skip_n):
        for i in ob.copy():
            ox = i[0]
            oy = i[1]
            dx = traj[ii, 0] - ox
            dy = traj[ii, 1] - oy

            r = math.sqrt(dx**2 + dy**2)

            if r <= config.robot_radius:
                return float("Inf")  # collision

            if minr >= r:
                minr = r

    return 1.0 / minr


# Calculate goal cost via Pythagorean distance to robot
def calc_to_goal_cost(traj, config):
    
    # If-Statements to determine negative vs positive goal/trajectory position
    # traj[-1,0] is the last predicted X coord position on the trajectory
    if (config.goalX >= 0 and traj[-1,0] < 0):
        dx = config.goalX - traj[-1,0]
    elif (config.goalX < 0 and traj[-1,0] >= 0):
        dx = traj[-1,0] - config.goalX
    else:
        dx = abs(config.goalX - traj[-1,0])
    
    # traj[-1,1] is the last predicted Y coord position on the trajectory
    if (config.goalY >= 0 and traj[-1,1] < 0):
        dy = config.goalY - traj[-1,1]
    elif (config.goalY < 0 and traj[-1,1] >= 0):
        dy = traj[-1,1] - config.goalY
    else:
        dy = abs(config.goalY - traj[-1,1])

    # print("dx, dy", dx, dy)
    cost = math.sqrt(dx**2 + dy**2)
    # print("Cost: ", cost)
    return cost


def calc_veg_cost(traj, config):
    # print("Trajectory end-points wrt odom", traj[-1, 0], traj[-1, 1])

    # Convert traj points to robot frame
    x_end_odom = traj[-1, 0]
    y_end_odom = traj[-1, 1]

    # Trajectory approx mid-points
    x_mid_odom = traj[math.floor(len(traj)/2), 0]
    y_mid_odom = traj[math.floor(len(traj)/2), 1]

    x_end_rob = (x_end_odom - config.x)*math.cos(config.th) + (y_end_odom - config.y)*math.sin(config.th)
    y_end_rob = -(x_end_odom - config.x)*math.sin(config.th) + (y_end_odom - config.y)*math.cos(config.th)
    x_mid_rob = (x_mid_odom - config.x)*math.cos(config.th) + (y_mid_odom - config.y)*math.sin(config.th)
    y_mid_rob = -(x_mid_odom - config.x)*math.sin(config.th) + (y_mid_odom - config.y)*math.cos(config.th)


    # int() and floor() behave differently with -ve numbers. int() is symmetric. 
    # cm_col = config.costmap_shape[0]/2 - math.floor(y_end_rob/config.costmap_resolution)
    # cm_row = config.costmap_shape[1]/2 - math.floor(x_end_rob/config.costmap_resolution)
    cm_col = config.costmap_shape[0]/2 - int(y_end_rob/config.costmap_resolution)
    cm_row = config.costmap_shape[1]/2 - int(x_end_rob/config.costmap_resolution)

    cm_mid_col = config.costmap_shape[0]/2 - int(y_mid_rob/config.costmap_resolution)
    cm_mid_row = config.costmap_shape[1]/2 - int(x_mid_rob/config.costmap_resolution)


    # !!! NOTE !!!: IN COSTMAP, VALUES SHOULD BE ACCESSED AS (ROW,COL). FOR VIZ, IT SHOULD BE (COL, ROW)! 
    # Sanity Check: Drawing end and mid points
    # config.costmap_rgb = cv2.circle(config.costmap_rgb, (int(cm_col), int(cm_row)), 1, (255, 255, 255), 1)
    # config.costmap_rgb = cv2.circle(config.costmap_rgb, (int(cm_mid_col), int(cm_mid_row)), 1, (0, 255, 0), 1)
    
    # print("Value at end-point = ", config.costmap_baselink[int(cm_row), int(cm_col)])
    # print("Max and min of costmap: ", np.max(config.costmap_baselink), np.min(config.costmap_baselink))

    # Cost which only considers trajectory end point
    # veg_cost = config.costmap_baselink_low[int(cm_row), int(cm_col)]
    
    # Cost which considers trajectory mid point and end point
    veg_cost = config.costmap_baselink_low[int(cm_row), int(cm_col)] + config.costmap_baselink_low[int(cm_mid_row), int(cm_mid_col)]

    return veg_cost



def draw_traj(config, traj, color):
    traj_array = np.asarray(traj)
    x_odom_list = np.asarray(traj_array[:, 0])
    y_odom_list = np.asarray(traj_array[:, 1])

    # print(x_odom_list.shape)

    x_rob_list, y_rob_list = odom_to_robot(config, x_odom_list, y_odom_list)
    cm_col_list, cm_row_list = robot_to_costmap(config, x_rob_list, y_rob_list)

    costmap_traj_pts = np.array((cm_col_list.astype(int), cm_row_list.astype(int))).T
    # print(costmap_traj_pts) 

    costmap_traj_pts = costmap_traj_pts.reshape((-1, 1, 2))
    config.costmap_rgb = cv2.polylines(config.costmap_rgb, [costmap_traj_pts], False, color, 1)
    
    return config.costmap_rgb




# NOTE: x_odom and y_odom are numpy arrays
def odom_to_robot(config, x_odom, y_odom):
    
    # print(x_odom.shape[0])
    x_rob_odom_list = np.asarray([config.x for i in range(x_odom.shape[0])])
    y_rob_odom_list = np.asarray([config.y for i in range(y_odom.shape[0])])

    x_rob = (x_odom - x_rob_odom_list)*math.cos(config.th) + (y_odom - y_rob_odom_list)*math.sin(config.th)
    y_rob = -(x_odom - x_rob_odom_list)*math.sin(config.th) + (y_odom - y_rob_odom_list)*math.cos(config.th)
    # print("Trajectory end-points wrt robot:", x_rob, y_rob)

    return x_rob, y_rob


def robot_to_costmap(config, x_rob, y_rob):

    costmap_shape_list_0 = [config.costmap_shape[0]/2 for i in range(y_rob.shape[0])]
    costmap_shape_list_1 = [config.costmap_shape[1]/2 for i in range(x_rob.shape[0])]

    y_list = [math.floor(y/config.costmap_resolution) for y in y_rob]
    x_list = [math.floor(x/config.costmap_resolution) for x in x_rob]

    cm_col = np.asarray(costmap_shape_list_0) - np.asarray(y_list)
    cm_row = np.asarray(costmap_shape_list_1) - np.asarray(x_list)
    # print("Costmap coordinates of end-points: ", (int(cm_row), int(cm_col)))

    return cm_col, cm_row


# Begin DWA calculations
def dwa_control(x, u, config, ob):
    # Dynamic Window control

    dw = calc_dynamic_window(x, config)

    u = calc_final_input(x, u, dw, config, ob)

    return u


# Determine whether the robot has reached its goal
def atGoal(config, x):
    # check at goal
    if math.sqrt((x[0] - config.goalX)**2 + (x[1] - config.goalY)**2) <= config.robot_radius:
        return True
    return False


def is_robot_stuck(config):

    # Condition for robot being stuck
    # NOTE: This condition may need to be changed to change in position or orientation
    
    # print("Robot's stuck locations: ", config.stuck_locations)
    # print("Robot's okay locations: ", config.okay_locations)
    # print("DWA Action: ", config.min_u)
    # print("Robot's current vel: ", config.v_x, config.w_z)
    
    if ((not config.pursuing_safe_loc) and (config.min_u != [0, 0] and config.min_u != []) and (abs(config.v_x) <= 0.05 and abs(config.w_z) <= 0.05)):
        config.stuck_count = config.stuck_count + 1
    else:
        config.stuck_count = 0

    if (config.stuck_count > 15):
        print("Robot could be stuck!")
        if (([math.floor(config.x), math.floor(config.y)] not in config.stuck_locations) and ([math.floor(config.x), math.floor(config.y)] not in config.okay_locations)): 
            # Stuck locations will only have integer coordinates. The "resolution" of the list is 1 meter.
            # Store stuck location
            config.stuck_locations.append([math.floor(config.x), math.floor(config.y)]) 

        return True # Stuck_status
    
    else:
        if (([math.floor(config.x), math.floor(config.y)] not in config.okay_locations) and ([math.floor(config.x), math.floor(config.y)] not in config.stuck_locations)): 
            # Okay locations will only have integer coordinates. The "resolution" of the list is 1 meter.
            # Store stuck location
            config.okay_locations.append([math.floor(config.x), math.floor(config.y)])

            # Experimental!
            # if (len(config.okay_locations) > 5 and config.happend_once == False):
            #     print("Collected 5 points. Stuck status = True!")
            #     config.happend_once = True
            #     return True

        return False

        

def recover(config):
    speed = Twist()

    config.pursuing_safe_loc = True

    x_odom = config.okay_locations[-2][0]
    y_odom = config.okay_locations[-2][1]

    # Convert the goal locations wrt robot frame. The error will simply be the goals.
    error_x = (x_odom - config.x)*math.cos(config.th) + (y_odom - config.y)*math.sin(config.th)
    error_y = -(x_odom - config.x)*math.sin(config.th) + (y_odom - config.y)*math.cos(config.th)

    print("(Recovery Point) --- (RobX, RobY) --- (Error X, Error Y) ")
    print(x_odom, y_odom, config.x, config.y, error_x, error_y)

    # Proportional gain
    k_p = 0.5
    vel_x = k_p * error_x
    vel_y = k_p * error_y

    # Note: This velocity assignment is for Spot cos it can move laterally
    # For a differential drive robot, use difference in angle and use it to compute w
    speed.linear.x = vel_x
    speed.linear.y = vel_y
    speed.angular.z = 0.0

    print(vel_x, vel_y)

    if (error_x < 0.5 and error_y < 0.5):
        print("Reached Safe Location!")
        config.pursuing_safe_loc = False
        config.stuck_status = False

        # Wait for 5 secs
        # time.sleep(5)

    return speed




def main():
    print(__file__ + " start!!")
    
    config = Config()
    obs = Obstacles()

    subOdom = rospy.Subscriber("/spot/odometry", Odometry, config.assignOdomCoords)
    subLaser = rospy.Subscriber("/scan", LaserScan, obs.assignObs, config)
    subGoal = rospy.Subscriber('/target/position', Twist, config.target_callback)

    # Costmap callbacks
    rospy.Subscriber("/high/move_base/local_costmap/costmap", OccupancyGrid, config.high_costmap_callback)
    rospy.Subscriber("/mid/move_base/local_costmap/costmap", OccupancyGrid, config.mid_costmap_callback)
    rospy.Subscriber("/low/move_base/local_costmap/costmap", OccupancyGrid, config.low_costmap_callback)
    
    # subCostmap = rospy.Subscriber("/low/move_base/local_costmap/costmap", OccupancyGrid, config.costmap_callback)
    subVegClassification = rospy.Subscriber("/vegetation_classes", Float32MultiArray, config.classification_callback)

    choice = input("Publish? 1 or 0")
    if(int(choice) == 1):
        pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        print("Publishing to cmd_vel")
    else:
        pub = rospy.Publisher("/dont_publish", Twist, queue_size=1)
        print("Not publishing!")

    speed = Twist()
    
    # initial state [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x = np.array([config.x, config.y, config.th, 0.0, 0.0])
    
    # initial linear and angular velocities
    u = np.array([0.0, 0.0])


    # runs until terminated externally
    while not rospy.is_shutdown():

        # config.stuck_status = is_robot_stuck(config)
        config.stuck_status = False

        # Initial
        if config.goalX == 0.0006 and config.goalY == 0.0006:
            # print("Initial condition")
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            x = np.array([config.x, config.y, config.th, 0.0, 0.0])
        
        # Pursuing but not reached the goal
        elif (atGoal(config,x) == False): 

            # Checking if robot is stuck
            if (config.stuck_status == True or config.pursuing_safe_loc == True):
                # Publish velocities accordingly
                speed = recover(config)
            
            else:
                u = dwa_control(x, u, config, obs.obst)

                x[0] = config.x
                x[1] = config.y
                x[2] = config.th
                x[3] = u[0]
                x[4] = u[1]
                speed.linear.x = x[3]
                speed.angular.z = x[4]


        # If at goal then stay there until new goal published
        else:
            print("Goal reached!")
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            x = np.array([config.x, config.y, config.th, 0.0, 0.0])
        
        pub.publish(speed)
        config.r.sleep()

    cv2.destroyAllWindows()



if __name__ == '__main__':
    rospy.init_node('dwa_costmap')
    main()