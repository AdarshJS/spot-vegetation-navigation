#!/usr/bin/env python

# This is inference code v2.
# Crops some x% off the top of the full-sized image and divides the rest into 2x3 patches.

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

# import the necessary packages
import roslib
import rospy
import cv2
import math
from sensor_msgs.msg import Image, Imu, CompressedImage, JointState, PointCloud2
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

from pyimagesearch.siamese_network import build_siamese_model
from pyimagesearch import config2 as config
from pyimagesearch import utils2 as utils
from pyimagesearch import metrics
from pyimagesearch import mobilenetv3
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np
# import tensorflow as tf
import time
import argparse

import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')



class Data_Subscriber:
	def __init__(self):
		self.bridge = CvBridge()
		self.img_topic_name = "/camera/color/image_raw/compressed" #/camera/color/image_raw
		self.velodyne_topic_name = "/velodyne_points"
		self.odom_topic_name = "/spot/odometry"  
		self.classifier_topic_name = '/vegetation_classes'

		# choice = input("Batch Length 12 or 24 ?")
		self.batch_length =  12 #int(choice)

		self.crop_top = 0.25

		# dist = input("Enter image distance threshold.")
		# self.distance_thresh = float(dist)

		self.distance_thresh = 0.4

		# Topic names
		self.image_sub = rospy.Subscriber(self.img_topic_name, CompressedImage, self.img_callback, queue_size=1,buff_size=2**25)
		self.classifier_pub = rospy.Publisher(self.classifier_topic_name, Float32MultiArray,queue_size=10)

		self.image_dims = (160, 90) #(240, 135) # (320, 180)
		self.costmap_shape = (200, 200)
		self.costmap_resolution = 0.05

		#loading reference images for each class from a folder
		reference_img_folder = '/home/spotcore/catkin_ws/src/spot-vegetation-navigation/siamese_network/reference_imgs/aug_demo_ref_imgs'
		self.classifier_msg = Float32MultiArray()

		self.ref_1 = utils.load_reference_imgs(reference_img_folder,'bushes_2',self.image_dims)[:,:,:3]
		self.ref_2 = utils.load_reference_imgs(reference_img_folder,'grass_1',self.image_dims)[:,:,:3]
		self.ref_3 = utils.load_reference_imgs(reference_img_folder,'grass_2',self.image_dims)[:,:,:3]
		self.ref_4 = utils.load_reference_imgs(reference_img_folder,'tree_1',self.image_dims)[:,:,:3]
		self.ref_5 = utils.load_reference_imgs(reference_img_folder,'grass_3',self.image_dims)[:,:,:3]

		self.ref_6 = utils.load_reference_imgs(reference_img_folder,'tree_2',self.image_dims)[:,:,:3]
		self.ref_7 = utils.load_reference_imgs(reference_img_folder,'tree_3',self.image_dims)[:,:,:3]

		# print(len(self.ref_5))
		# print(len(self.ref_3))

		## load the model from disk

		MODEL_FOLDER = '/home/spotcore/catkin_ws/src/spot-vegetation-navigation/siamese_network/aug_demo_models/'

		model_choice = input("Load model number ? ")

		if int(model_choice) == 1:
			model_path = MODEL_FOLDER + "1_175_iters"
		elif int(model_choice) == 2:
			model_path = MODEL_FOLDER + "2_190_iters_w_equalization"
		elif int(model_choice) == 3:
			model_path = MODEL_FOLDER + "3_110_iters"
		elif int(model_choice) == 4:
			model_path = MODEL_FOLDER + "4_80_iters_w_equalization"
		elif int(model_choice) == 5:
			model_path = MODEL_FOLDER + "5_100_iters_split_dataset"
		elif int(model_choice) == 6:
			model_path = MODEL_FOLDER + "6_225_iters"
		elif int(model_choice) == 7:
			model_path = MODEL_FOLDER + "60_iters_w_equalization"

		print(".... loading siamese model...")
		print("Model path :", model_path)
		self.model = load_model(model_path,custom_objects={'contrastive_loss': metrics.contrastive_loss},compile=False)
		print("Starting Inference Code v2")
		

	def img_callback(self,img_data):
		try:
		# 	cv_image = self.bridge.imgmsg_to_cv2(img_data, "bgr8") # for uncompressed images
			cv_image  = self.bridge.compressed_imgmsg_to_cv2(img_data) # for compressed images
 
		except CvBridgeError as e:
			print(e)

		# #visualize image
		# cv2.imshow('dst',cv_image)
		# cv2.waitKey(3)

		print("Sequence No: ",img_data.header.seq)
		(self.rows,self.cols,channels) = cv_image.shape
		(height, width, _) = cv_image.shape


		# Generate pairs from current image and the reference images. 
		ref_current_imgs = self.generate_pairs(cv_image)
		# print(type(ref_current_imgs))

		image = cv_image


		# prediction using model inference
		time1= time.time()
		predictions = self.model_inference(ref_current_imgs)
		time2= time.time()
		inference_time = time2 -time1
		print("Inference time:", inference_time)
		# print("printing predictions........")
		# print(predictions)


		if self.batch_length == 12:

			pred1 = predictions[0:2]
			pred2 = predictions[2:4]
			pred3 = predictions[4:6]
			pred4 = predictions[6:8]
			pred5 = predictions[8:10]
			pred6 = predictions[10:]

			quad_1,class_val_1 = self.category_selector_v2(pred1)
			quad_2,class_val_2 = self.category_selector_v2(pred2)
			quad_3,class_val_3 = self.category_selector_v2(pred3)
			quad_4,class_val_4 = self.category_selector_v2(pred4)
			quad_5,class_val_5 = self.category_selector_v2(pred5)
			quad_6,class_val_6 = self.category_selector_v2(pred6)

			self.classifier_msg.data = [class_val_1, min(pred1), class_val_2, min(pred2), class_val_3, min(pred3),class_val_4, min(pred4), class_val_5, min(pred5), class_val_6, min(pred6)]
			print("Classifier msg:",self.classifier_msg.data)

		elif self.batch_length == 24:

			pred1 = predictions[0:4]
			pred2 = predictions[4:8]
			pred3 = predictions[8:12]
			pred4 = predictions[12:16]
			pred5 = predictions[16:20]
			pred6 = predictions[20:]

			pred1_idx = np.where(pred1 == pred1.min())[0]
			pred2_idx = np.where(pred2 == pred2.min())[0]
			pred3_idx = np.where(pred3 == pred3.min())[0]
			pred4_idx = np.where(pred4 == pred4.min())[0]
			pred5_idx = np.where(pred5 == pred5.min())[0]
			pred6_idx = np.where(pred6 == pred6.min())[0]

			# print("indices:",pred1_idx,pred2_idx,pred3_idx,pred4_idx)

			quad_1,class_val_1 = self.category_selector(pred1,pred1_idx)
			quad_2,class_val_2 = self.category_selector(pred2,pred2_idx)
			quad_3,class_val_3 = self.category_selector(pred3,pred3_idx)
			quad_4,class_val_4 = self.category_selector(pred4,pred4_idx)
			quad_5,class_val_5 = self.category_selector(pred5,pred5_idx)
			quad_6,class_val_6 = self.category_selector(pred6,pred6_idx)
			# print("class Vals:",[class_val_1,class_val_2,class_val_3,class_val_4])

			self.classifier_msg.data = [class_val_1,pred1[pred1_idx][0],class_val_2,pred2[pred2_idx][0],class_val_3,pred3[pred3_idx][0],class_val_4,pred4[pred4_idx][0],class_val_5,pred5[pred5_idx][0],class_val_6,pred6[pred6_idx][0]]
			print("Classifier msg:",self.classifier_msg.data)

		#publish class of each quadrant and it's confidence [class1,confidence1,...] 0-Non-pliable 1-Pliable
		self.classifier_pub.publish(self.classifier_msg)

		# VISUALIZATION
		# font = cv2.FONT_HERSHEY_SIMPLEX
		# fontScale = 1
		# color = (0, 255, 255)
		# thickness = 3
		# image = cv2.putText(image, quad_1, (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
		# image = cv2.putText(image, quad_2, (int(self.cols/2),50), font, fontScale, color, thickness, cv2.LINE_AA)
		# image = cv2.putText(image, quad_3, (50, int(self.rows/2)), font, fontScale, color, thickness, cv2.LINE_AA)
		# image = cv2.putText(image, quad_4, (int(self.cols/2),int(self.rows/2)), font, fontScale, color, thickness, cv2.LINE_AA)
		



	def generate_pairs(self,cv_image):
		# generate pairs from reference images and the current image

		(height, width, _) = cv_image.shape

		# Crop some top portion of the full-sized image
		image = cv_image[int(height*self.crop_top):height, 0:width]



		# # Crop into six patches (1-top left, 2- top center 3-top right, 4- bottom left, 5- bottom center, 6- bottom right)       
		crop_1 = cv_image[0:int(height/2), 0:int(width/3)]
		crop_2 = cv_image[0:int(height/2), int(width/3)+1:int(2*width/3)]
		crop_3 = cv_image[0:int(height/2), int(2*width/3)+1:width]

		crop_4 = cv_image[int(height/2)+1:height, 0:int(width/3)]
		crop_5 = cv_image[int(height/2)+1:height, int(width/3)+1:int(2*width/3)]
		crop_6 = cv_image[int(height/2)+1:height, int(2*width/3)+1:width]	


		# #resize and normalize each quadrant image
		crop_1_img = np.array(cv2.resize(crop_1, self.image_dims, interpolation = cv2.INTER_AREA))/255.0
		crop_2_img = np.array(cv2.resize(crop_2, self.image_dims, interpolation = cv2.INTER_AREA))/255.0
		crop_3_img = np.array(cv2.resize(crop_3, self.image_dims, interpolation = cv2.INTER_AREA))/255.0
		crop_4_img = np.array(cv2.resize(crop_4, self.image_dims, interpolation = cv2.INTER_AREA))/255.0
		crop_5_img = np.array(cv2.resize(crop_5, self.image_dims, interpolation = cv2.INTER_AREA))/255.0
		crop_6_img = np.array(cv2.resize(crop_6, self.image_dims, interpolation = cv2.INTER_AREA))/255.0


		ref_current_imgs= [] # reference and current image pair list variable initialization

		# # For batch size 4 predictions
		# ref_current_imgs.append([crop_1_img,self.ref_img])
		# ref_current_imgs.append([crop_2_img,self.ref_img])
		# ref_current_imgs.append([crop_3_img,self.ref_img])
		# ref_current_imgs.append([crop_4_img,self.ref_img])
		
		if self.batch_length == 24:
			# For batch size 24 predictions
			ref_current_imgs.append([crop_1_img,self.ref_1])
			ref_current_imgs.append([crop_1_img,self.ref_5])
			ref_current_imgs.append([crop_1_img,self.ref_3])
			ref_current_imgs.append([crop_1_img,self.ref_7])

			ref_current_imgs.append([crop_2_img,self.ref_1])
			ref_current_imgs.append([crop_2_img,self.ref_5])
			ref_current_imgs.append([crop_2_img,self.ref_3])
			ref_current_imgs.append([crop_2_img,self.ref_7])

			ref_current_imgs.append([crop_3_img,self.ref_1])
			ref_current_imgs.append([crop_3_img,self.ref_5])
			ref_current_imgs.append([crop_3_img,self.ref_3])
			ref_current_imgs.append([crop_3_img,self.ref_6])

			ref_current_imgs.append([crop_4_img,self.ref_1])
			ref_current_imgs.append([crop_4_img,self.ref_5])
			ref_current_imgs.append([crop_4_img,self.ref_3])	
			ref_current_imgs.append([crop_4_img,self.ref_6])

			ref_current_imgs.append([crop_5_img,self.ref_1])
			ref_current_imgs.append([crop_5_img,self.ref_5])
			ref_current_imgs.append([crop_5_img,self.ref_3])	
			ref_current_imgs.append([crop_5_img,self.ref_6])

			ref_current_imgs.append([crop_6_img,self.ref_1])
			ref_current_imgs.append([crop_6_img,self.ref_5])
			ref_current_imgs.append([crop_6_img,self.ref_3])	
			ref_current_imgs.append([crop_6_img,self.ref_6])

		elif self.batch_length == 12:
			# For batch size 8 predictions with only grass
			ref_current_imgs.append([crop_1_img,self.ref_5])
			ref_current_imgs.append([crop_1_img,self.ref_3])

			ref_current_imgs.append([crop_2_img,self.ref_5])
			ref_current_imgs.append([crop_2_img,self.ref_3])

			ref_current_imgs.append([crop_3_img,self.ref_5])
			ref_current_imgs.append([crop_3_img,self.ref_3])

			ref_current_imgs.append([crop_4_img,self.ref_5])
			ref_current_imgs.append([crop_4_img,self.ref_3])

			ref_current_imgs.append([crop_5_img,self.ref_5])
			ref_current_imgs.append([crop_5_img,self.ref_3])

			ref_current_imgs.append([crop_6_img,self.ref_5])
			ref_current_imgs.append([crop_6_img,self.ref_3])

		ref_current_imgs= np.array(ref_current_imgs)
		# print(ref_current_imgs.shape())

		# # Sanity check
		# print("Dims of input img pairs:", np.array(ref_current_imgs).shape)
		# print("Dims of one element in the pair: ", ref_current_imgs[0, 0].shape)

		# cv2.imshow('cv_img_1', crop_1_img)
		# cv2.imshow('cv_img_2', crop_2_img)
		# cv2.imshow('cv_img_3', crop_3_img)
		# cv2.imshow('cv_img_4', crop_4_img)
		# cv2.imshow('cv_img_5', crop_5_img)
		# cv2.imshow('cv_img_6', crop_6_img)
		# cv2.imshow('Top cropped image', image)

		# cv2.waitKey(3)

		return ref_current_imgs

	def model_inference(self,ref_current_imgs):
		#model inference for predictions

		# # assigning current and reference image pairs as input img pairs
		imgA = ref_current_imgs[:,0] #pairTrain[0, 0] #Input(shape=config.IMG_SHAPE)
		imgB = ref_current_imgs[:,1]  #Input(shape=config.IMG_SHAPE)


		# print("imgA shape:",imgA.shape)
		# print("imgB shape:",imgB.shape)

		# imgA = tf.squeeze(imgA, axis=0)
		# imgB = tf.squeeze(imgB, axis=0)

		# # # add a batch dimension to both images
		# imgA = tf.expand_dims(imgA, axis=0)
		# imgB = tf.expand_dims(imgB, axis=0)

		# siamese model to make predictions on the image pair,
		# indicating whether or not the images belong to the same class
		preds = self.model.predict([imgA, imgB])

		# print("printing predictions........")
		# print(preds)

		return preds[:,0]

	def category_selector(self,pred_array,idx):

		quadrant =''
		class_val = 2

		if idx == 0:
			quadrant = 'Non-pliable'#'Bushes'
			class_val = 0
		elif idx == 1:
			class_val = 1
			if pred_array[idx] <= self.distance_thresh:
				quadrant = 'Pliable'#'Dense Grass'
			else:
				quadrant = 'Non-Pliable Grass'
		elif idx == 2:
			class_val = 1
			if pred_array[idx] <= self.distance_thresh:
				quadrant = 'Pliable'#'Sparse Grass'
			else:
				quadrant = 'Non-Pliable Sparse Grass'
		elif idx == 3:
			class_val = 0
			quadrant = 'Non-pliable' #'Trees'

		# Q1_cols,Q1_rows = self.robot_to_costmap(np.array([5.18,2.59]),np.array([0.84,0]))
		# Q2_cols,Q2_rows = self.robot_to_costmap(np.array([5.18,2.59]),np.array([0,-0.84]))
		# Q3_cols,Q3_rows = self.robot_to_costmap(np.array([2.59,0.9]),np.array([0.84,0]))
		# Q4_cols,Q4_rows = self.robot_to_costmap(np.array([2.59,0.9]),np.array([0,-0.84]))

		# print("Q1 coords:",Q1_cols,Q1_rows)
		# print("Q2 coords:",Q2_cols,Q2_rows)
		# print("Q3 coords:",Q3_cols,Q3_rows)
		# print("Q4 coords:",Q4_cols,Q4_rows)

		return quadrant,class_val

	def category_selector_v2(self,pred_array):
		class_val = 2
		quadrant =''
		if pred_array[0] < self.distance_thresh or pred_array[1] < self.distance_thresh:
			class_val = 1
			quadrant = 'Pliable'
		else:
			class_val = 0
			quadrant = 'Non-Pliable'

		return quadrant, class_val


def main(args):
	ic = Data_Subscriber()
	rospy.init_node('data_subscriber', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)


