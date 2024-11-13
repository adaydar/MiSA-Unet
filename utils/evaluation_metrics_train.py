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

from utils.model import *
from utils.loader import *
from utils.config import *
from utils.loss_functions import * 

def check_accuracy(loader, model, loss_fn,device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = [0,0,0,0,0]
    model.eval()
    VOE=[0,0,0,0,0]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = (model(x))
            y1 = y.float().unsqueeze(1)
            loss_f = loss_fn(preds,y1)
            preds = nn.Softmax(dim=1)(preds)
            pred_class = torch.argmax(preds,dim=1)

            y = torch.moveaxis(y, (1, 2), (2, 3))
            y_class = torch.argmax(y,dim=1)
            num_correct += (pred_class == y_class).sum()
            num_pixels += torch.numel(pred_class)

            
            for i in range(5):
                dice_score[i] += (2 * (preds[:, i, :, :] * y[:, i, :, :]).sum()) / (
                    (preds[:, i, :, :] + y[:, i, :, :]).sum() + 1e-8
                )
                VOE[i] +=1- ((2 * (preds[:, i, :, :] * y[:, i, :, :]).sum()) / ((preds[:, i, :, :] + y[:, i, :, :]).sum() + 1e-8)/(2-((2 * (preds[:, i, :, :] * y[:, i, :, :]).sum()) / ((preds[:, i, :, :] + y[:, i, :, :]).sum() + 1e-8))))
                


    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    
    print(f"Dice score for Femoral Bone: {dice_score[1]/len(loader)}")
    print(f"Dice score for Femoral Cartillage: {dice_score[2]/len(loader)}")
    print(f"Dice score for Tibial Bone: {dice_score[3]/len(loader)}")
    print(f"Dice score for Tibial Cartillage: {dice_score[4]/len(loader)}")
    print(f"Average Dice score: {(dice_score[1]+dice_score[2]+dice_score[3]+dice_score[4])/(4* len(loader))}")
    print(f"VOE for Femoral Bone: {VOE[1]}")
    print(f"VOE for Femoral Cartillage: {VOE[2]}")
    print(f"VOE for Tibial Bone: {VOE[3]}")
    print(f"VOE for Tibial Cartillage: {VOE[4]}") 

    model.train()
    return loss_f

    
def save_predictions_as_imgs(
    loader, model, folder=SAVE_PATH+"val_pred/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = (model(x))
            preds = nn.Softmax(dim=1)(preds)
            pred_class = torch.argmax(preds,dim=1)
            pred_class = pred_class/4
        
        for i in range(pred_class.shape[0]):
            pred_indi = pred_class[i:i+1,:,:]
            torchvision.utils.save_image(pred_indi.unsqueeze(1), f"{folder}/pred_{idx*8+i}.png")
            
        y = torch.moveaxis(y, (1, 2), (2, 3))
        y_class = torch.argmax(y,dim=1)
        y_class = y_class/4
        
        for i in range(y_class.shape[0]):
            y_indi = y_class[i:i+1,:,:]
            torchvision.utils.save_image(y_indi.unsqueeze(1), f"{folder}{idx*8+i}.png")

    model.train()
