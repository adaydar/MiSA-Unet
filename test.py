import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import time
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import sys
import pandas as pd
import csv

from utils.model import *
from utils.loader_test import *
from utils.config import *
from utils.loss_functions import * 
from utils.evaluation_metrics_test import *

images = os.listdir(TEST_IMG_DIR)
images2=(os.listdir(TEST_IMG_DIR)).sort()
dict1 = {'original_list':images,'model_list':images2}
images1 = pd.DataFrame.from_dict(dict1)
images1.to_csv(SAVE_PATH+"test_image_list.csv")

        
def test_fn(loader, model, loss_fn, scaler):

    for idx, (data, targets) in tqdm(enumerate(loader), total=len(loader)):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)



def main():
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = MiSA_Unet().to(DEVICE)
    loss_fn = combined_loss

    test_loader = get_test_dataloader(
        images,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL_TEST:
        load_checkpoint(torch.load(SAVE_PATH+"model.pth", map_location=DEVICE), model)
        print(f"Loaded Model Successfully")
    
    dict2={}
    scaler = torch.cuda.amp.GradScaler()
    start = time.time()
    sys.stdout = open(SAVE_PATH+"TestLog.txt", "w")
    test_fn(test_loader, model, loss_fn, scaler)
    dc_1,dc_2,dc_3,dc_4, EVOE_1,EVOE_2, EVOE_3, EVOE_4 = check_accuracy(test_loader, model, device=DEVICE)
    dict_2= {'FB':dc_1,"FC":dc_2,"TB":dc_3,"TC":dc_4}
    dict_4= {'FB':EVOE_1,"FC":EVOE_2,"TB":EVOE_3,"TC":EVOE_4}
    DC_score = pd.DataFrame(dict_2)
    VOE_score = pd.DataFrame(dict_4)
  
    DC_score.to_csv(SAVE_PATH+"DC.csv")
    VOE_score.to_csv(SAVE_PATH+"VOE.csv")
    print("Images saved", flush=True)
    end = time.time()
    print(f"Total time: {end-start} seconds" )
    sys.stdout.close()

if __name__ == "__main__":
    main()
