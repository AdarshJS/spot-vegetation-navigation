# USAGE
# python train_siamese_network.py

# import the necessary packages
from pyimagesearch.siamese_network import build_siamese_model
from pyimagesearch import config2 as config
from pyimagesearch import utils3 as utils
from pyimagesearch import metrics
from pyimagesearch import mobilenetv3
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import numpy as np
import hickle as hkl



import time


numClasses = 4 # pliable-dense, pliable-ground, trees, non-pliable-dense

hidden_dim = 1000

# NOTE: (width, height) format only for reshaping PIL images in make_pairs(). If values change, also change in config2.py 
image_dims = (160, 90) #(240, 135) # # (320, 180)

# NOTE: UNCOMMENT IF NEW TRAINING AND TESTING PAIRS ARE NEEDED

# # Folders which contain training images
# pliable_dense_train = "/media/asathyam/Media/spot-veg/dataset_new/sorted_split/train/pliable-dense-train"
# pliable_ground_train = "/media/asathyam/Media/spot-veg/dataset_new/sorted_split/train/pliable-ground-train"
# non_pliable_dense_train = "/media/asathyam/Media/spot-veg/dataset_new/sorted_split/train/non-pliable-dense-train"
# trees_train = "/media/asathyam/Media/spot-veg/dataset_new/sorted_split/train/tree-train"
# train_folder_list = [pliable_dense_train, pliable_ground_train, non_pliable_dense_train, trees_train]

# # Folders which contain test images
# pliable_dense_test = "/media/asathyam/Media/spot-veg/dataset_new/sorted_split/test/pliable-dense-test"
# pliable_ground_test = "/media/asathyam/Media/spot-veg/dataset_new/sorted_split/test/pliable-ground-test"
# non_pliable_dense_test = "/media/asathyam/Media/spot-veg/dataset_new/sorted_split/test/non-pliable-dense-test"
# trees_test = "/media/asathyam/Media/spot-veg/dataset_new/sorted_split/test/tree-test"
# test_folder_list = [pliable_dense_test, pliable_ground_test, non_pliable_dense_test, trees_test]

# # prepare the positive and negative pairs 
# print("[INFO] preparing positive and negative pairs...")
# t1 = time.time()
# (pairTrain, labelTrain) = utils.make_pairs(train_folder_list, numClasses, image_dims)
# t2 = time.time()
# (pairTest, labelTest) = utils.make_pairs(test_folder_list, numClasses, image_dims)
# t3 = time.time()

# print("Time to create training pairs: ", t2 - t1)
# print("Time to create test pairs: ", t3 - t2)

# # Save as .npy file for future use
# print("Saving pairTrain and pairTest")
# np.save('pairTrain.npy', pairTrain)
# np.save('labelTrain.npy', labelTrain)
# np.save('pairTest.npy', pairTest)
# np.save('labelTest.npy', labelTest)


## TRAINING #################################################################

npy_folder = '/media/kasun/Media/VeRN/new_dataset_v2' #"/home/kasun/spot-vegetation-navigation/dataset_new"
# Load training data

for iteration in range(5):

	if iteration != 0:
		model = load_model(config.MODEL_PATH,custom_objects={'contrastive_loss': metrics.contrastive_loss},compile=False)

	print("[INFO]: Loading data!")
	pairTrain = np.array(hkl.load(npy_folder + "/pairTrain"+ str(iteration) +'.hkl')) #np.load(npy_folder + "/pairTrain.npy")
	labelTrain = np.array(hkl.load(npy_folder + "/labelTrain"+ str(iteration) +'.hkl'))  # np.load(npy_folder + "/labelTrain.npy")

	# Load test data
	pairTest = np.array(hkl.load(npy_folder + "/pairTest" +'.hkl'))  #np.load(npy_folder + "/pairTest.npy")
	labelTest = np.array(hkl.load(npy_folder + "/labelTest" +'.hkl')) #np.load(npy_folder + "/labelTest.npy")

	# Sanity check
	print("Dims of pairTrain:", pairTrain.shape)
	print("Dims of one element in pairTrain: ", pairTrain[0, 0].shape)

	# configure the siamese network
	print("[INFO] building siamese network...")
	imgA = Input(shape=config.IMG_SHAPE)
	imgB = Input(shape=config.IMG_SHAPE)

	# Check the type of this
	featureExtractor = mobilenetv3.MobileNetV3(type="feature", input_shape=config.IMG_SHAPE, classes_number=hidden_dim)
	# featureExtractor = build_siamese_model(config.IMG_SHAPE)
	featsA = featureExtractor(imgA)
	featsB = featureExtractor(imgB) # this featsB for all classes can be saved so that it need not be computed again

	# finally, construct the siamese network
	distance = Lambda(utils.euclidean_distance)([featsA, featsB])
	outputs = Dense(1, activation="sigmoid")(distance)
	model = Model(inputs=[imgA, imgB], outputs=outputs)

	# compile the model
	print("[INFO] compiling model with contrastive loss...")
	# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	model.compile(loss=metrics.contrastive_loss, optimizer="adam")
	# model.compile(loss=metrics.contrastive_loss, optimizer=Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999))

	# train the model
	print("[INFO] training model...")
	history = model.fit(
		[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
		validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
		batch_size=config.BATCH_SIZE, 
		epochs=config.EPOCHS)

	# # serialize the model to disk
	print("[INFO] saving siamese model...")
	model.save(config.MODEL_PATH)

# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)