import torch

LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
BATCH_SIZE = 150
test_batch_size = 1
NUM_EPOCHS = 100
NUM_WORKERS = 0
IMAGE_HEIGHT = 150  
IMAGE_WIDTH = 150  
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_TEST = True
SPLIT = 0.3

TRAIN_IMG_DIR =  "./Data2/Training/train_img_1"

TRAIN_MASK_DIR = "./Data2/Training/train_mask_1" 

TEST_IMG_DIR = "./Data2/Testing/test_img_4" 

TEST_MASK_DIR = "./Data2/Testing/test_mask_4"


SAVE_PATH = "./MiSA-Unet/"

def save_checkpoint(state, filename=SAVE_PATH+"model.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
