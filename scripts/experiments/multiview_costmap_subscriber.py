#!/usr/bin/env python

# This code subscribes to 3 costmap topics, sums them, marks the robot and tall obstacles 
# on the costmap. These functions were eventually integrated into dwa_costmap.py

import rospy
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.path import Path

from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid, Odometry
from tf.transformations import euler_from_quaternion

from PIL import Image

import sys
# OpenCV
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

class Multiview_Costmap():
    def __init__(self):
        print("Initializing!")
        self.costmap_shape = (200, 200)
        self.costmap_resolution = 0.05
        
        self.costmap_baselink_high = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.costmap_baselink_mid = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.costmap_baselink_low = np.zeros(self.costmap_shape, dtype=np.uint8)


    def robot_marker(self, costmap_gray):

        # Marking the robot
        costmap_rgb = cv2.cvtColor(costmap_gray,cv2.COLOR_GRAY2RGB)
        rob_x = int(costmap_rgb.shape[0]/2)
        rob_y = int(costmap_rgb.shape[1]/2)
        marked_image = cv2.circle(costmap_rgb, (rob_x, rob_y), 2, (0, 255, 0), 2)

        return marked_image
         
    
    def resize(self, image, scale):
        # resize image
        scale_percent = scale # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        return resized


    def tall_obstacle_marker(self, rgb_image, centers):
        rgb_image[centers[:, 0], centers[:, 1], 0] = 0
        rgb_image[centers[:, 0], centers[:, 1], 1] = 0
        rgb_image[centers[:, 0], centers[:, 1], 2] = 255

        return rgb_image


    def remove_points(self, rgb_image, centers):
        print(rgb_image.shape)
        rgb_image[centers[:, 0], centers[:, 1], 0] = 0
        rgb_image[centers[:, 0], centers[:, 1], 1] = 255
        rgb_image[centers[:, 0], centers[:, 1], 2] = 0

        return rgb_image



    def high_costmap_callback(self, data):
        
        print("Received high-view costmap!")

        # Subscribe the low-view costmap
        costmap_2d_high = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        costmap_2d_high = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        costmap_2d_high = np.rot90(np.fliplr(costmap_2d_high), 1, (1, 0))

        cm_image_high = Image.fromarray(np.uint8(costmap_2d_high))

        # Subscribe to odometry/filtered
        odom_data = rospy.wait_for_message('/spot/odometry', Odometry, timeout=10)
        orientation_q = odom_data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

        yaw_deg = yaw*180/math.pi

        cm_baselink_pil_high = cm_image_high.rotate(-yaw_deg)
        self.costmap_baselink_high = np.array(cm_baselink_pil_high)

        rgb = self.robot_marker(self.costmap_baselink_high)
        resized = self.resize(rgb, 200)

        # cv2.imshow('costmap_high', resized)
        # cv2.waitKey(3)


    def mid_costmap_callback(self, data):
        
        print("Received mid-view costmap!")

        # Subscribe the mid-view costmap
        costmap_2d_mid = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        costmap_2d_mid = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        costmap_2d_mid = np.rot90(np.fliplr(costmap_2d_mid), 1, (1, 0))

        cm_image_mid = Image.fromarray(np.uint8(costmap_2d_mid))

        # Subscribe to odometry/filtered
        odom_data = rospy.wait_for_message('/spot/odometry', Odometry, timeout=10)
        orientation_q = odom_data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

        yaw_deg = yaw*180/math.pi

        cm_baselink_pil_mid = cm_image_mid.rotate(-yaw_deg)
        self.costmap_baselink_mid = np.array(cm_baselink_pil_mid)

        rgb = self.robot_marker(self.costmap_baselink_mid)
        resized = self.resize(rgb, 200)

        # cv2.imshow('costmap_mid', resized)
        # cv2.waitKey(3)



    def low_costmap_callback(self, data):
        
        print("Received low-view costmap!")

        # Subscribe the low-view costmap
        costmap_2d_low = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        costmap_2d_low = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        costmap_2d_low = np.rot90(np.fliplr(costmap_2d_low), 1, (1, 0))

        cm_image_low = Image.fromarray(np.uint8(costmap_2d_low))

        # Subscribe to odometry/filtered
        odom_data = rospy.wait_for_message('/spot/odometry', Odometry, timeout=10)
        orientation_q = odom_data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

        yaw_deg = yaw*180/math.pi

        cm_baselink_pil_low = cm_image_low.rotate(-yaw_deg)
        self.costmap_baselink_low = np.array(cm_baselink_pil_low)

        rgb = self.robot_marker(self.costmap_baselink_low)
        resized = self.resize(rgb, 200)

        self.costmap_sum()

        # cv2.imshow('costmap_low', resized)
        # cv2.waitKey(3)

    def costmap_sum(self):
        
        test_cm = self.costmap_baselink_low + self.costmap_baselink_mid + self.costmap_baselink_high

        resized = test_cm #self.resize(test_cm, 100)

        # where there is a greater than 50% chance of obstacles
        # obs_high = np.argwhere(self.costmap_baselink_high > 50)  
        # obs_mid = np.argwhere(self.costmap_baselink_mid > 50)
        # obs_low = np.argwhere(self.costmap_baselink_low > 50)

        # obs_high_set = set((tuple(i) for i in obs_high))
        # obs_mid_set = set((tuple(i) for i in obs_mid))
        # obs_low_set = set((tuple(i) for i in obs_low))
        # obs_mid_high = obs_high_set.intersection(obs_mid_set) #np.array(list(obs_high_set.intersection(obs_mid_set)))
        # obs_low_mid_high = np.array(list(obs_low_set.intersection(obs_mid_high)))
        
        # print("Obstacle points in high: ", obs_high)
        # print("Obstacle points in mid: ", obs_mid)
        # print(obs_mid_high)
        
        obs_low_mid_high = np.argwhere(resized > 150)
        

        robot_marked = self.robot_marker(resized)
        if(obs_low_mid_high.shape[0] != 0):
            # final = self.remove_points(robot_marked, obs_low_mid_high)
            final = self.tall_obstacle_marker(robot_marked, obs_low_mid_high)
        else:
            final = robot_marked

        final_resized = self.resize(final, 200)
        

        cv2.imshow('costmap_sum', final_resized)
        cv2.waitKey(3)





    


def listener():

    rospy.init_node('local_costmap_subscriber', anonymous=True)

    multiview_cm_obj = Multiview_Costmap()

    rospy.Subscriber("/high/move_base/local_costmap/costmap", OccupancyGrid, multiview_cm_obj.high_costmap_callback)
    rospy.Subscriber("/mid/move_base/local_costmap/costmap", OccupancyGrid, multiview_cm_obj.mid_costmap_callback)
    rospy.Subscriber("/low/move_base/local_costmap/costmap", OccupancyGrid, multiview_cm_obj.low_costmap_callback)

    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    listener()
