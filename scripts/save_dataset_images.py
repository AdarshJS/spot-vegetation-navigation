
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

        self.directory = '/media/asathyam/Media/spot-veg/1_dataset/umd_multi_6'
        self.type = 'umd_multi_6'
        os.chdir(self.directory)
        self.iter = 0


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print('received image of type: "%s"' % ros_data.format)

        # Conversion to CV2
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:

        # Resize image
        scale_percent = 50 # percent of original size
        width = int(image_np.shape[1] * scale_percent / 100)
        height = int(image_np.shape[0] * scale_percent / 100)
        dim = (width, height)

        
         
        # resize image
        # print("Image dimensions:", dim)
        resized = cv2.resize(image_np, dim, interpolation = cv2.INTER_AREA)

        # Save image
        # print("Current Dir:", os.listdir(self.directory))
        print(self.iter)
        file_name = self.type + '_' + str(self.iter) + '.png'
        cv2.imwrite(file_name, resized)
        self.iter = self.iter + 1
        
        
        cv2.imshow('cv_img', resized)
        cv2.waitKey(3)

        # time.sleep(0.5)

        #### Create CompressedIamge ####
        # msg = CompressedImage()
        # msg.header.stamp = rospy.Time.now()
        # msg.format = "jpeg"
        # msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # # Publish new image
        # self.image_pub.publish(msg)
        
        #self.subscriber.unregister()

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