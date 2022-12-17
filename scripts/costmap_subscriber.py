#!/usr/bin/env python

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


def costmap_callback(data):
    
    print("Received local costmap!")

    costmap_2d = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
    costmap_2d = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
    costmap_2d = np.rot90(np.fliplr(costmap_2d), 1, (1, 0))

    cm_image = Image.fromarray(np.uint8(costmap_2d))

    # Subscribe to odometry/filtered
    odom_data = rospy.wait_for_message('/odometry/filtered', Odometry, timeout=10)
    orientation_q = odom_data.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

    yaw_deg = yaw*180/math.pi

    cm_baselink_pil = cm_image.rotate(-yaw_deg)
    costmap_baselink = np.array(cm_baselink_pil)

    cv2.imshow('cv_img', costmap_baselink)
    cv2.waitKey(3)


def listener():

    rospy.init_node('local_costmap_subscriber', anonymous=True)

    rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, costmap_callback)

    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    listener()
