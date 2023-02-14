import shutil
import os
import math
import random

if __name__ == "__main__":
	train_folder = "/media/asathyam/Media/spot-veg/5_dataset_final/train/bushes_train"
	test_folder = "/media/asathyam/Media/spot-veg/5_dataset_final/test/bushes_test"

	test_percentage = 0.25

	num_images = len([name for name in os.listdir(train_folder)])
	num_test_images = int(math.floor(test_percentage * num_images))
	print("Number of images: ", num_test_images)

	for i in range(num_test_images):

		# Pick a random image
		rand_file = random.choice(os.listdir(train_folder))
		print(rand_file)

		full_path = train_folder + "/" + rand_file
		dest = shutil.move(full_path, test_folder)
		# break
