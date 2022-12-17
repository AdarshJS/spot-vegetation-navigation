# import the necessary packages
import os

# specify the shape of the inputs for our network
IMG_SHAPE = (28, 28, 1)

# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 10

# define the path to the base output directory
BASE_OUTPUT = "output"

# use the base output path to derive the path to the serialized
# model along with training history plot
# MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
MODEL_PATH = '/home/asathyam/1_spot_grass/compare-images-siamese-networks/output/siamese_model/'
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])