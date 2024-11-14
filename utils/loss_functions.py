import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.model import *

class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.conv1 = conv_block(5, 16)
        self.conv2 = conv_block(16, 16)
        self.conv3 = conv_block(16, 16)
        self.conv4 = conv_block(16, 5)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        return x4, x3, x2, x1
        
class CycleNet(nn.Module):
    def __init__(self):
        super(CycleNet, self).__init__()
        self.b_block = block()
    def forward(self, x):
        t1,t2,t3,t4 = self.b_block(x)
        return t1,t2,t3,t4  

       
def SR_loss(preds,GT):
       Loss_1 = 0.0
       Loss_2 = 0.0
       Loss_3 = 0.0
       Loss_4 = 0.0
       #preds = preds.to(DEVICE)
       GT = torch.squeeze(GT)
       LossL1 = nn.L1Loss()
       loss_net = CycleNet().to(DEVICE)
       t1,t2,t3,t4 = loss_net(preds)
       tg1,tg2,tg3,tg4 = loss_net(GT)
       Loss_1=LossL1(tg1,t1)
       Loss_2=LossL1(tg2,t2)
       Loss_3=LossL1(tg3,t3)
       Loss_4=LossL1(tg4,t4)
       return  Loss_1,Loss_2,Loss_3,Loss_4

def combined_loss(preds, y, eps=1e-7):
    preds = nn.Softmax(dim=1)(preds)
    #y1 = y
    y = torch.moveaxis(y, (2, 3), (3, 4))
    dice_score = [0,0,0,0,0]
    y1 = torch.squeeze(y)
    y1 = y1.permute(0,1,2,3)
    preds1= preds.permute(0,1,2,3)
    BCE = F.cross_entropy(preds1,y1,weight = torch.tensor([0.01,0.1,0.27,0.12,0.5]).to(DEVICE))
    BAL1,BAL2,BAL3,BAL4 = SR_loss(preds,y)
    for i in range(5):
        inputs = preds[:, i, :, :]
        targets = y[:, 0, i, :, :]
        dice_score[i] += (2 * (preds[:, i, :, :] * y[:, 0, i, :, :]).sum()) / (
            (preds[:, i, :, :] + y[:, 0, i, :, :]).sum() + eps
        )        
    X = 1 - (dice_score[0]*0.01+dice_score[1]*0.1+dice_score[2]*0.27+dice_score[3]*0.12+dice_score[4]*0.5)
    BAL = (BAL1*0.1+BAL2*0.2+BAL3*0.3+BAL4*0.4)
    Y = 0.7*(X+BCE)+0.3*(BAL)  
    return Y
