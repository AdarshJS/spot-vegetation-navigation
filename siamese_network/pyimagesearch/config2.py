# import the necessary packages
import os

# specify the shape of the inputs for our network
# IMG_SHAPE = (28, 28, 1)
# IMG_SHAPE = (180, 320, 3) # (height, width, channels)
IMG_SHAPE = (90, 160, 3) # (height, width, channels)
# IMG_SHAPE = (135, 240, 3) # (height, width, channels)

# specify the batch size and number of epochs
BATCH_SIZE = 32
EPOCHS = 75 # 50 or 100

# define the path to the base output directory
BASE_OUTPUT = str(EPOCHS)

# use the base output path to derive the path to the serialized
# model along with training history plot
# MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
MODEL_PATH = './aug_demo_models/' + str(EPOCHS) #'/media/asathyam/Media/spot-veg/siamese_network/output/siamese_model'
PLOT_PATH = os.path.sep.join([MODEL_PATH, "plot.png"])
