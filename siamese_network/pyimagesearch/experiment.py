import os, os.path
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import time
import numpy as np


def main():
    folder = '/media/asathyam/Media/spot-veg/dataset_new/sorted/non-pliable-dense'
    image_dims = (160, 90)
    for idxA in range(2):#range(len([name for name in os.listdir(folder)])):
			
		current_image_name = os.listdir(folder)[idxA]
		currentImage = Image.open(folder + "/" + current_image_name)
		currentImage = currentImage.resize(image_dims)
		currentImage.show()
		currentImage = ImageOps.equalize(currentImage, mask = None)
		currentImage.show()

		currentImage = np.array(currentImage)
		currentImage = currentImage / 255.0

if __name__ == "__main__":
    main()