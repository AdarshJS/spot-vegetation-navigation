#!/usr/bin/env python

# This is inference code v1.

# import the necessary packages
import roslib
import rospy
# import cv2
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
import tensorflow as tf
import time
import argparse

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


class Data_Subscriber:
	def __init__(self):
		self.bridge = CvBridge()
		self.img_topic_name = "/camera/color/image_raw/compressed" #/camera/color/image_raw
		self.velodyne_topic_name = "/velodyne_points"
		self.odom_topic_name = "/spot/odometry"  
		self.classifier_topic_name = '/vegetation_classes'

		choice = input("Batch Length 8 or 16 ?")
		self.batch_length = int(choice)

		# Topic names
		self.image_sub = rospy.Subscriber(self.img_topic_name, CompressedImage, self.img_callback, queue_size=1,buff_size=2**25)
		self.classifier_pub = rospy.Publisher(self.classifier_topic_name, Float32MultiArray,queue_size=10)

		self.image_dims = (160, 90) #(240, 135) #(160, 90) # (320, 180)
		self.costmap_shape = (200, 200)
		self.costmap_resolution = 0.05

		#loading reference images for each class from a folder
		reference_img_folder = './reference_imgs'
		self.classifier_msg = Float32MultiArray()

		self.ref_1 = utils.load_reference_imgs(reference_img_folder,'bushes_2',self.image_dims)[:,:,:3]
		self.ref_2 = utils.load_reference_imgs(reference_img_folder,'grass_dense_1',self.image_dims)[:,:,:3]
		self.ref_3 = utils.load_reference_imgs(reference_img_folder,'grass_sparse_1',self.image_dims)[:,:,:3]
		self.ref_4 = utils.load_reference_imgs(reference_img_folder,'trees_4',self.image_dims)[:,:,:3]
		self.ref_5 = utils.load_reference_imgs(reference_img_folder,'dried_grass1',self.image_dims)[:,:,:3]

		self.ref_6 = utils.load_reference_imgs(reference_img_folder,'trees_3_quad',self.image_dims)[:,:,:3]
		self.ref_7 = utils.load_reference_imgs(reference_img_folder,'trees_4_quad',self.image_dims)[:,:,:3]
# 
		# self.ref_img =self.ref_5[:,:,:3]
		# print('Ref Image dims:', self.ref_img.shape)

		# load the model from disk
		print(".... loading siamese model...")
		print(config.MODEL_PATH)
		self.model = load_model(config.MODEL_PATH,custom_objects={'contrastive_loss': metrics.contrastive_loss},compile=False)
		

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
		# print(" camera img (rows,cols,channels)",self.rows,self.cols,channels)

		# # current image is the entire image for now.TODO: split to 4 quadrants and pair separately
		# currentImage = cv2.resize(cv_image, self.image_dims, interpolation = cv2.INTER_AREA) #cv_image.resize(self.image_dims)
		# currentImage = np.array(currentImage)

		# # print("current image shape:",currentImage.shape)
		# currentImage = (currentImage) / 255.0 # Normalize between 0-1s
		# currentImage = np.array(currentImage)


		# generate pairs from current image and the reference images. 4 pairs with 4 rereference images from 4 classes and the current image
		ref_current_imgs = self.generate_pairs(cv_image)

		image = cv_image


		# prediction using model inference

		time1= time.time()
		predictions = self.model_inference(ref_current_imgs)
		time2= time.time()
		inference_time = time2 -time1
		print("Inference time:", inference_time)
		# print("printing predictions........")
		# print(predictions)


		if self.batch_length == 8:

			pred1 = predictions[0:2]
			pred2 = predictions[2:4]
			pred3 = predictions[4:6]
			pred4 = predictions[6:]

			quad_1,class_val_1 = self.category_selector_v2(pred1)
			quad_2,class_val_2 = self.category_selector_v2(pred2)
			quad_3,class_val_3 = self.category_selector_v2(pred3)
			quad_4,class_val_4 = self.category_selector_v2(pred4)

			self.classifier_msg.data = [class_val_1,min(pred1),class_val_2,min(pred2),class_val_3,min(pred3),class_val_4,min(pred4)]
			print("Classifier msg:",self.classifier_msg.data)

		elif self.batch_length == 16:

			pred1 = predictions[0:4]
			pred2 = predictions[4:8]
			pred3 = predictions[8:12]
			pred4 = predictions[12:]

			pred1_idx = np.where(pred1 == pred1.min())[0]
			pred2_idx = np.where(pred2 == pred2.min())[0]
			pred3_idx = np.where(pred3 == pred3.min())[0]
			pred4_idx = np.where(pred4 == pred4.min())[0]

			# print("indices:",pred1_idx,pred2_idx,pred3_idx,pred4_idx)

			quad_1,class_val_1 = self.category_selector(pred1,pred1_idx)
			quad_2,class_val_2 = self.category_selector(pred2,pred2_idx)
			quad_3,class_val_3 = self.category_selector(pred3,pred3_idx)
			quad_4,class_val_4 = self.category_selector(pred4,pred4_idx)
			# print("class Vals:",[class_val_1,class_val_2,class_val_3,class_val_4])

			self.classifier_msg.data = [class_val_1,pred1[pred1_idx][0],class_val_2,pred2[pred2_idx][0],class_val_3,pred3[pred3_idx][0],class_val_4,pred4[pred4_idx][0]]
			print("Classifier msg:",self.classifier_msg.data)

		#publish class of each quadrant and it's confidence [class1,confidence1,...] 0-Non-pliable 1-Pliable
		self.classifier_pub.publish(self.classifier_msg)

		font = cv2.FONT_HERSHEY_SIMPLEX
		fontScale = 1
		color = (0, 255, 255)
		thickness = 3
		image = cv2.putText(image, quad_1, (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
		image = cv2.putText(image, quad_2, (int(self.cols/2),50), font, fontScale, color, thickness, cv2.LINE_AA)
		image = cv2.putText(image, quad_3, (50, int(self.rows/2)), font, fontScale, color, thickness, cv2.LINE_AA)
		image = cv2.putText(image, quad_4, (int(self.cols/2),int(self.rows/2)), font, fontScale, color, thickness, cv2.LINE_AA)
		

		# # projecting the nearby segmented ground region to ground plane
		# # All points are in format [cols, rows]
		# pt_A = [506, 198] # LEFT TOP
		# pt_B = [0, 720] #LEFT BOTTOM
		# pt_C = [1280, 720] # RIGHT BOTTOM
		# pt_D = [806, 188] # RIGHT TOP

		# width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
		# width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
		# maxWidth = max(int(width_AD), int(width_BC))
		 
		# height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
		# height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
		# maxHeight = max(int(height_AB), int(height_CD))


		# input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])

		# output_pts = np.float32([[0, 0],
		#                         [0, maxHeight - 1],
		#                         [maxWidth - 1, maxHeight - 1],
		#                         [maxWidth - 1, 0]])


		# # Compute the perspective transform M
		# M = cv2.getPerspectiveTransform(input_pts,output_pts)

		# projected_out = cv2.warpPerspective(cv_image,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
		
		# dim_projected = (600, 450)
		# projected_out = cv2.resize(projected_out, dim_projected, interpolation = cv2.INTER_AREA)

		#visualize image
		# cv2.imshow('Projected',projected_out)
		# cv2.imshow('Predictions',image)
		# cv2.waitKey(1)



	def generate_pairs(self,cv_image):
		# generate pairs from reference images and the current image

		# loading a dummy image from a folder as the current image for testing. need to be replaced with the image subscriber
		# current_img = utils.load_reference_imgs(self.reference_img_folder,'trees_2',image_dims)

		(height, width, _) = cv_image.shape
		# print("height,width :",height, width)

		# Crop into four quadrants (1-top left, 2-top right, 3- bottom left, 4- bottom right)       
		crop_1 = cv_image[0:int(height/2), 0:int(width/2)]
		crop_2 = cv_image[0:int(height/2), int(width/2)+1:width]
		crop_3 = cv_image[int(height/2)+1:height, 0:int(width/2)]
		crop_4 = cv_image[int(height/2)+1:height, int(width/2)+1:width]	


		#resize and normalize each quadrant image
		crop_1_img = np.array(cv2.resize(crop_1, self.image_dims, interpolation = cv2.INTER_AREA))/255.0
		crop_2_img = np.array(cv2.resize(crop_2, self.image_dims, interpolation = cv2.INTER_AREA))/255.0
		crop_3_img = np.array(cv2.resize(crop_3, self.image_dims, interpolation = cv2.INTER_AREA))/255.0
		crop_4_img = np.array(cv2.resize(crop_4, self.image_dims, interpolation = cv2.INTER_AREA))/255.0

		ref_current_imgs= [] # reference and current image pair list variable initialization

		# # For batch size 4 predictions
		# ref_current_imgs.append([crop_1_img,self.ref_img])
		# ref_current_imgs.append([crop_2_img,self.ref_img])
		# ref_current_imgs.append([crop_3_img,self.ref_img])
		# ref_current_imgs.append([crop_4_img,self.ref_img])
		
		if self.batch_length == 16:
			# For batch size 16 predictions
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

		elif self.batch_length == 8:
			# For batch size 8 predictions with only grass
			# ref_current_imgs.append([crop_1_img,self.ref_1])
			ref_current_imgs.append([crop_1_img,self.ref_5])
			ref_current_imgs.append([crop_1_img,self.ref_3])
			# ref_current_imgs.append([crop_1_img,self.ref_4])

			# ref_current_imgs.append([crop_2_img,self.ref_1])
			ref_current_imgs.append([crop_2_img,self.ref_5])
			ref_current_imgs.append([crop_2_img,self.ref_3])
			# ref_current_imgs.append([crop_2_img,self.ref_4])

			# ref_current_imgs.append([crop_3_img,self.ref_1])
			ref_current_imgs.append([crop_3_img,self.ref_5])
			ref_current_imgs.append([crop_3_img,self.ref_3])
			# ref_current_imgs.append([crop_3_img,self.ref_4])

			# ref_current_imgs.append([crop_4_img,self.ref_1])
			ref_current_imgs.append([crop_4_img,self.ref_5])
			ref_current_imgs.append([crop_4_img,self.ref_3])	
			# ref_current_imgs.append([crop_4_img,self.ref_4])

		ref_current_imgs= np.array(ref_current_imgs)

		# # Sanity check
		# print("Dims of input img pairs:", np.array(ref_current_imgs).shape)
		# print("Dims of one element in the pair: ", ref_current_imgs[0, 0].shape)

		# cv2.imshow('cv_img_1', crop_1_img)
		# cv2.imshow('cv_img_2', crop_2_img)
		# cv2.imshow('cv_img_3', crop_3_img)
		# cv2.imshow('cv_img_4', crop_4_img)
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
			if pred_array[idx] <=0.4:
				quadrant = 'Pliable'#'Dense Grass'
			else:
				quadrant = 'Non-Pliable Grass'
		elif idx == 2:
			class_val = 1
			if pred_array[idx] <=0.4:
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
		if pred_array[0] < 0.4 or pred_array[1] < 0.4:
			class_val = 1
			quadrant = 'Pliable'
		else:
			class_val = 0
			quadrant = 'Non-Pliable'

		return quadrant, class_val


	def robot_to_costmap(self, x_rob, y_rob):

	    costmap_shape_list_0 = [self.costmap_shape[0]/2 for i in range(y_rob.shape[0])]
	    costmap_shape_list_1 = [self.costmap_shape[1]/2 for i in range(x_rob.shape[0])]

	    y_list = [math.floor(y/self.costmap_resolution) for y in y_rob]
	    x_list = [math.floor(x/self.costmap_resolution) for x in x_rob]

	    cm_col = np.asarray(costmap_shape_list_0) - np.asarray(y_list)
	    cm_row = np.asarray(costmap_shape_list_1) - np.asarray(x_list)
	    # print("Costmap coordinates of end-points: ", (int(cm_row), int(cm_col)))

	    return cm_col, cm_row

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


