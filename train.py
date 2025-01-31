from torch.utils.data import Dataset
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np

from utils.model import *
from utils.loader import *
from utils.config import *
from utils.loss_functions import * 
from utils.evaluation_metrics_train import *
  
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    Loss_cum = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss1 = loss_fn(predictions, targets)
            loss2, loss_net1 = SRL_cartilage(predictions, targets, data)
            loss = loss1+loss2 
            Loss_cum += loss.item()  
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # update tqdm loop
        loop.set_postfix(loss=loss.item())
        Train_loss = Loss_cum/len(loader)
    print(Train_loss)

def main():
    train_transform = A.Compose(
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

    val_transforms = A.Compose(
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

    model = MtRA_Unet().to(DEVICE)
    loss_fn = combined_loss 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        train_images_path,
        val_images_path,    
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(SAVE_PATH+"model.pth"), model)
        
    max_yet = check_accuracy(val_loader, model, loss_fn, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
  
    start = time.time()
    sys.stdout = open(SAVE_PATH+"Log.txt", "w")
    for epoch in range(NUM_EPOCHS):
        #print(epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        
        cur_acc = check_accuracy(val_loader, model, loss_fn, device=DEVICE)
        
        if cur_acc < max_yet:
            # print some examples to a folder
            max_yet = cur_acc
            save_checkpoint(checkpoint)
            print("Checkpoint saved", flush=True)
            save_predictions_as_imgs(
                val_loader, model, folder=SAVE_PATH+"val_pred/", device=DEVICE
            )
            print("Images saved", flush=True)
        end = time.time()
        print(f"Total time: {end-start} seconds" )
    sys.stdout.close()

if __name__ == "__main__":
    main()
