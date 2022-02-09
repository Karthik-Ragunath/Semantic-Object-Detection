import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 100 # number of epochs to train for

# training images and XML files directory
TRAIN_DIR = "data/out_rgb"
# validation images and XML files directory
VALID_DIR = "data/out_rgb"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# classes: 0 index is reserved for background
CLASSES = [
    '0', '1', '2', '3'
]
NUM_CLASSES = 4

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs