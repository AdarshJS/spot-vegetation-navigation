# import the necessary packages
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np

import os, os.path
from pathlib import Path
import math
import random

import sys
# OpenCV
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from PIL import Image

def make_pairs(folder_list):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []

	numClasses = 4 # grass-dense, grass-sparse, trees, bushes
	image_dims = (320, 180)

	for i in range(numClasses):
		current_folder = folder_list[i]
		for idxA in range(len([name for name in os.listdir(folder_list[i])])):
			# currentImage = random.choice(os.listdir(current_folder))
			
			current_image_name = os.listdir(current_folder)[idxA]
			# currentImage = cv2.imread(current_folder + "/" + current_image_name)
			currentImage = Image.open(current_folder + "/" + current_image_name)
			currentImage = currentImage.resize(image_dims)

			pos_image_name = random.choice(os.listdir(current_folder)) # pick a random image from the same folder
			# posImage = cv2.imread(current_folder + "/" + pos_image_name)
			posImage = Image.open(current_folder + "/" + pos_image_name)
			posImage = posImage.resize(image_dims)

			# prepare a positive pair and update the images and labels lists, respectively
			pairImages.append([currentImage, posImage]) # appending a tuple
			pairLabels.append([1])

			folder_idx = np.arange(numClasses) #[0, 1, 2, 3]
			
			negIdx = random.choice([j for j in folder_idx if j != i])
			neg_image_name = random.choice(os.listdir(folder_list[negIdx]))
			# negImage = cv2.imread(folder_list[negIdx] + "/" + neg_image_name)
			negImage = Image.open(folder_list[negIdx] + "/" + neg_image_name)
			negImage = negImage.resize(image_dims)

			# prepare a negative pair of images and update our lists
			pairImages.append([currentImage, negImage])
			pairLabels.append([0])

			# Sanity check
			pos_pair = np.concatenate((currentImage, posImage), axis=0)
			neg_pair = np.concatenate((currentImage, negImage), axis=0)

			# Convert PIL to Opencv image
			pos_pair = pos_pair[:, :, ::-1].copy() 
			neg_pair = neg_pair[:, :, ::-1].copy() 

			# Visualization
			# cv2.imshow('Positive pair', pos_pair)
			# cv2.imshow('Negative pair', neg_pair)
			# cv2.waitKey(500)


	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))


def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors

	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)

	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)


# For test purposes
# if __name__ == "__main__":

# 	grass_dense = "/media/asathyam/Media/spot-veg/5_dataset_final/train/grass_dense_train"
# 	grass_sparse = "/media/asathyam/Media/spot-veg/5_dataset_final/train/grass_dense_train"
# 	bushes = "/media/asathyam/Media/spot-veg/5_dataset_final/train/bushes_train"
# 	trees = "/media/asathyam/Media/spot-veg/5_dataset_final/train/trees_train"
	
# 	folder_list = [grass_dense, grass_sparse, bushes, trees]

# 	make_pairs(folder_list)

	# print(len([name for name in os.listdir(folder_list[3])])) # finds the number of objects in a dir
