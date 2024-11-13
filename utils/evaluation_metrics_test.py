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
from skimage import measure  
from scipy.spatial.distance import directed_hausdorff

from utils.model import *
from utils.loader_test import *
from utils.config import *
from utils.loss_functions import * 

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = [0,0,0,0,0]   
    VOE = [0,0,0,0,0]
    counter_1 = 0
    dc1=[]
    dc2=[]
    dc3=[]
    dc4=[]
    EVOE1 = []
    EVOE2 = []
    EVOE3 = []
    EVOE4 = []
    Ind_DC=[0,0,0,0,0]
    Ind_VOE = [0,0,0,0,0]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = (model(x))
            preds = nn.Softmax(dim=1)(preds)
            pred_class = torch.argmax(preds,dim=1)

            y = torch.moveaxis(y, (1, 2), (2, 3))
            y_class = torch.argmax(y,dim=1)
            #ACC = (pred_class == y_class).sum()
            #PIX = torch.numel(pred_class)
        
            
            Fem = torch.numel(pred_class[pred_class==1])
            Fem_cart = torch.numel(pred_class[pred_class==2])
            Tib = torch.numel(pred_class[pred_class==3])
            Tib_cart = torch.numel(pred_class[pred_class==4])
            
            if Fem>=300 and Tib>=300 and Fem_cart>=100 and Tib_cart>=100:
                    num_correct += (pred_class == y_class).sum()
                    num_pixels += torch.numel(pred_class)   
                    counter_1+=1
                    for i in range(5): 
                        #print(type(preds[:, i, :, :].sum()))
                        #print(preds[:, i, :, :].sum().shape)
                        Ind_DC[i]= (2 * (preds[:, i, :, :] * y[:, i, :, :]).sum()) / ((preds[:, i, :, :] + y[:, i, :, :]).sum() + 1e-8)  
                        Ind_VOE[i] = 1- ((2 * (preds[:, i, :, :] * y[:, i, :, :]).sum()) / ((preds[:, i, :, :] + y[:, i, :, :]).sum() + 1e-8)/(2-((2 * (preds[:, i, :, :] * y[:, i, :, :]).sum()) / ((preds[:, i, :, :] + y[:, i, :, :]).sum() + 1e-8))))

                                     
                        dice_score[i] += (2 * (preds[:, i, :, :] * y[:, i, :, :]).sum()) / ((preds[:, i, :, :] + y[:, i, :, :]).sum() + 1e-8)
                        VOE[i] +=1- ((2 * (preds[:, i, :, :] * y[:, i, :, :]).sum()) / ((preds[:, i, :, :] + y[:, i, :, :]).sum() + 1e-8)/(2-((2 * (preds[:, i, :, :] * y[:, i, :, :]).sum()) / ((preds[:, i, :, :] + y[:, i, :, :]).sum() + 1e-8))))

                        if i==1:
                          k1=Ind_DC[1].to(dtype=torch.float32).to('cpu').item()
                          d1 = Ind_VOE[1].to(dtype=torch.float32).to('cpu').item()
                          dc1.append(k1)
                          EVOE1.append(d1)

                        elif i==2:
                          k2 = Ind_DC[2].to(dtype=torch.float32).to('cpu').item()
                          d2 = Ind_VOE[2].to(dtype=torch.float32).to('cpu').item() 
                          dc2.append(k2)
                          EVOE2.append(d2)
                        
                        elif i==3:
                          k3 = Ind_DC[3].to(dtype=torch.float32).to('cpu').item()
                          d3 = Ind_VOE[3].to(dtype=torch.float32).to('cpu').item() 
                          dc3.append(k3)
                          EVOE3.append(d3)
                      
                        elif i==4:
                          k4 = Ind_DC[4].to(dtype=torch.float32).to('cpu').item()
                          d4 = Ind_VOE[4].to(dtype=torch.float32).to('cpu').item() 
                          dc4.append(k4)
                          EVOE4.append(d4)
                         
            else:
                    #ACC = 0
                    #PIX = 0
                    counter_1+=0
                    dice_score[1] += 0
                    dice_score[2] += 0
                    dice_score[3] += 0
                    dice_score[4] += 0
                    VOE [1] += 0
                    VOE [2] += 0
                    VOE [3] += 0
                    VOE [4] += 0

    dice_sc_1 = dice_score[1]/counter_1
    dice_sc_2 = dice_score[2]/counter_1
    dice_sc_3 = dice_score[3]/counter_1
    dice_sc_4 = dice_score[4]/counter_1
    Avearge_dice = (dice_score[1]+dice_score[2]+dice_score[3]+dice_score[4])/(4* counter_1)
    VOE_1 = VOE[1]/counter_1
    VOE_2 = VOE[2]/counter_1
    VOE_3 = VOE[3]/counter_1
    VOE_4 = VOE[4]/counter_1
  
    print(f"Got {num_correct}/{num_pixels} with acc { (num_correct/num_pixels)*100:.2f}")  
    print(f"Dice score for Femoral Bone: {dice_sc_1}")
    print(f"Dice score for Femoral Cartillage: {dice_sc_2}")
    print(f"Dice score for Tibial Bone: {dice_sc_3}")
    print(f"Dice score for Tibial Cartillage: {dice_sc_4}")
    print(f"Average Dice score: {Avearge_dice}")
    print(f"VOE for Femoral Bone: {VOE_1}")
    print(f"VOE for Femoral Cartillage: {VOE_2}")
    print(f"VOE for Tibial Bone: {VOE_3}")
    print(f"VOE for Tibial Cartillage: {VOE_4}")  

    return dc1,dc2,dc3,dc4, EVOE1,EVOE2, EVOE3, EVOE4
   
   
def save_predictions_as_imgs(
    loader, model, ground_folder, pred_folder, device="cuda"
):

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = (model(x))
            preds = nn.Softmax(dim=1)(preds)
            pred_class = torch.argmax(preds,dim=1)
            pred_class1 = pred_class/4
            
            Fem = torch.numel(pred_class[pred_class==1])
            Fem_cart = torch.numel(pred_class[pred_class==2])
            Tib = torch.numel(pred_class[pred_class==3])
            Tib_cart = torch.numel(pred_class[pred_class==4])
                 
        for i in range(pred_class1.shape[0]):
          if Fem>=300 and Tib>=300 and Fem_cart>=100 and Tib_cart>=100:
            pred_indi = pred_class1[i:i+1,:,:]
            torchvision.utils.save_image(pred_indi.unsqueeze(1), f"{pred_folder}/pred_{idx+i}.png")
            
        y = torch.moveaxis(y, (1, 2), (2, 3))
        y_class = torch.argmax(y,dim=1)
        y_class = y_class/4
       
        for i in range(y_class.shape[0]):
            y_indi = y_class[i:i+1,:,:]
            #torchvision.utils.save_image(y_indi.unsqueeze(1), f"{ground_folder}{idx+i}.png")
            

def calculate_hd(mask1, mask2):
    mask1_np = mask1.cpu().numpy()
    mask2_np = mask2.cpu().numpy()
    print(mask1_np.shape) 
    contours1, _ = measure.find_contours(mask1_np.astype(np.uint8), 0.5)
    contours2, _ = measure.find_contours(mask2_np.astype(np.uint8), 0.5)
    if len(contours1) == 0:
        return None
    elif len(contours2) == 0:
        return None    
    boundary1 = np.squeeze(contours1[0], axis=1)
    boundary2 = np.squeeze(contours2[0], axis=1)
    hd1 = directed_hausdorff(boundary1, boundary2)[0].max()
    hd2 = directed_hausdorff(boundary2, boundary1)[0].max()
    return max(hd1, hd2)

