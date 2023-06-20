# This code subscribes to image topic from a rosbag, uses every n-th image and saves patches that divide the 
# image into 2x3

# Python libs
import os
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

# import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE=False

class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        # self.image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size = 1)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.callback,  queue_size = 1)
        print("subscribed to /camera/image/compressed")

        # self.directory = '/media/asathyam/Media/spot-veg/dataset_new/2_small_cropped_2x3_unsorted'
        self.type = 'gq_may25_2'
        # os.chdir(self.directory)
        self.iter = 0
        self.image_number = 0
        self.skip = 15
        self.crop_top = 0.20


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print('received image of type: "%s"' % ros_data.format)

        if (self.image_number%self.skip == 0):
            
            self.image_number = self.image_number + 1

            # Conversion to CV2
            np_arr = np.fromstring(ros_data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
            orig_width = image_np.shape[1]
            orig_height = image_np.shape[0]
            # print("Width, height:", orig_width, orig_height)

            image = image_np[int(orig_height*self.crop_top):orig_height, 0:orig_width]
            (height, width, _) = image.shape

            # # Resize image
            # scale_percent = 50 # percent of original size
            # new_width = int(image_np.shape[1] * scale_percent / 100)
            # new_height = int(image_np.shape[0] * scale_percent / 100)
            # dim = (new_width, new_height)
            # # print("Image dimensions:", dim)
            # resized = cv2.resize(image_np, dim, interpolation = cv2.INTER_AREA)

            # # Crop into six patches (1-top left, 2- top center 3-top right, 4- bottom left, 5- bottom center, 6- bottom right)       
            crop_1 = image[0:int(height/2), 0:int(width/3)]
            crop_2 = image[0:int(height/2), int(width/3)+1:int(2*width/3)]
            crop_3 = image[0:int(height/2), int(2*width/3)+1:width]

            crop_4 = image[int(height/2)+1:height, 0:int(width/3)]
            crop_5 = image[int(height/2)+1:height, int(width/3)+1:int(2*width/3)]
            crop_6 = image[int(height/2)+1:height, int(2*width/3)+1:width]

            # # Save image patches
            # # print("Current Dir:", os.listdir(self.directory))
            # file_name = self.type + '_' + str(self.iter) + '.png'
            # cv2.imwrite(file_name, resized)

            folder_1 = "/media/asathyam/Media/spot-veg/dataset_new/3_small_cropped_2x3_unsorted/1"
            folder_2 = "/media/asathyam/Media/spot-veg/dataset_new/3_small_cropped_2x3_unsorted/2"
            folder_3 = "/media/asathyam/Media/spot-veg/dataset_new/3_small_cropped_2x3_unsorted/3"
            folder_4 = "/media/asathyam/Media/spot-veg/dataset_new/3_small_cropped_2x3_unsorted/4"
            folder_5 = "/media/asathyam/Media/spot-veg/dataset_new/3_small_cropped_2x3_unsorted/5"
            folder_6 = "/media/asathyam/Media/spot-veg/dataset_new/3_small_cropped_2x3_unsorted/6"

            print(self.iter)

            crop_1_name = folder_1 + "/" + self.type + '_' + str(self.iter) + "_1.png"
            crop_2_name = folder_2 + "/" + self.type + '_' + str(self.iter) + "_2.png"
            crop_3_name = folder_3 + "/" + self.type + '_' + str(self.iter) + "_3.png"
            crop_4_name = folder_4 + "/" + self.type + '_' + str(self.iter) + "_4.png"
            crop_5_name = folder_5 + "/" + self.type + '_' + str(self.iter) + "_5.png"
            crop_6_name = folder_6 + "/" + self.type + '_' + str(self.iter) + "_6.png"

            cv2.imwrite(crop_1_name, crop_1)
            cv2.imwrite(crop_2_name, crop_2)
            cv2.imwrite(crop_3_name, crop_3)
            cv2.imwrite(crop_4_name, crop_4)
            cv2.imwrite(crop_5_name, crop_5)
            cv2.imwrite(crop_6_name, crop_6)

            self.iter = self.iter + 1
            
            # Display Image
            # cv2.imshow('cv_img', image_np)
            # cv2.imshow('cv_img_1', crop_1)
            # cv2.imshow('cv_img_2', crop_2)
            # cv2.imshow('cv_img_3', crop_3)
            # cv2.imshow('cv_img_4', crop_4)
            # cv2.imshow('cv_img_5', crop_5)
            # cv2.imshow('cv_img_6', crop_6)
            cv2.imshow('Top cropped image', image)
            cv2.waitKey(3)

        else:
            self.image_number = self.image_number + 1


def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)