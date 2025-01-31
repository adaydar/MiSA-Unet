import torch
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

       
def SRL_overall(preds,GT):
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
    BAL1,BAL2,BAL3,BAL4 = SRL_overall(preds,y)
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
    
### Cartilage refinment #####
class block1(nn.Module):
    def __init__(self):
        super(block1, self).__init__()
        self.conv1 = conv_block(1, 16)
        self.conv2 = conv_block(16, 16)
        self.conv3 = conv_block(16, 16)
        self.self_attention = SelfAttention()  # Add self-attention module
        self.conv4 = conv_block(16, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        #x2 = self.self_attention(x2)  # Apply self-attention after the third convolution
        x3 = self.conv3(x2)
        x3 = self.self_attention(x3, x3)  # Apply self-attention after the third convolution
        x4 = self.conv4(x3)

        return x4

class CycleNet1(nn.Module):
    def __init__(self):
        super(CycleNet1, self).__init__()
        self.b_block1 = block1()
    def forward(self, x):
        t1 = self.b_block1(x)
        return t1 
        
def basic_commands_conversion(preds):
       pred_class = torch.argmax(preds,dim=1)
       pred_class = pred_class/4 
       pred_coarse_mask1 = pred_class.unsqueeze(1).cpu()
       return pred_coarse_mask1

        
def process_batch(pred_indi, thresholds, folder):

    batch_results = []
    num_batch = pred_indi.shape[0]
    
    # Loop through each image in the batch
    for k in range(num_batch):
        pred_indi_b = pred_indi[k,:,:,:]
        pred_indi_tensor = pred_indi_b.squeeze(0).cpu()  # Remove batch dimension if it exists

        # Apply thresholds and generate images
        thresholded_images = []
        for threshold in thresholds:
            thresholded_image = (pred_indi_tensor*255 > threshold).float()  # Threshold and scale to 255
            thresholded_image1 = thresholded_image.unsqueeze(dim=0)
            thresholded_images.append(thresholded_image)

        # Perform operations between thresholded images
        im_1, im_2, im_3 = thresholded_images[:3]  # Take the first three thresholded images
        im_4 = torch.clamp(im_3 - im_2, min=0)  # Subtract im_2 from im_3 and clamp values to ensure no negative values
        final_image = torch.clamp(im_1 + im_4, max=255)  # Add im_1 and im_4 and clamp to 255 to stay within range
        final_image = final_image.unsqueeze(dim=0)
        # Add the final tensor to results
        batch_results.append(final_image)    
    # Stack all tensors into a single batch tensor
    z = torch.stack(batch_results)
    return z
    
                      
def SRL_cartilage(pred_coarse_mask, gt_fine_mask, image, num_steps=2, noise_weight=0.1):

    pred_coarse_mask = basic_commands_conversion(pred_coarse_mask)
    pred_coarse_mask = process_batch(pred_coarse_mask, [200,160,80])
    
    gt_fine_mask = gt_fine_mask.squeeze(1)    
    gt_fine_mask = torch.moveaxis(gt_fine_mask, (1, 2), (2, 3))
    gt_fine_mask = basic_commands_conversion(gt_fine_mask)
    gt_fine_mask = process_batch(gt_fine_mask, [200,160,80])
        
    pred_coarse_mask = pred_coarse_mask.to(DEVICE)
    gt_fine_mask = gt_fine_mask.to(DEVICE)
    
    forward_loss = 0.0
    degraded_mask = gt_fine_mask.clone()
    
    # === Forward Diffusion Loss === #
    for step in range(num_steps):
        noise = torch.randn_like(degraded_mask) * noise_weight
        degraded_mask = torch.where(
            torch.rand_like(degraded_mask) > step / num_steps,
            degraded_mask + noise,
            pred_coarse_mask
        )
        forward_loss += F.mse_loss(degraded_mask, pred_coarse_mask)

    forward_loss /= num_steps

    # === Reverse Diffusion Loss === #
    reverse_loss = 0.0
    refined_mask = pred_coarse_mask.clone()

    for step in range(num_steps):
        input_data = refined_mask
        input_data = input_data.to(DEVICE)
        with autocast():
         loss_net1 = CycleNet1().to(DEVICE)
         refined_mask_step= loss_net1(input_data)                
         reverse_loss += F.mse_loss(refined_mask_step, gt_fine_mask)

    reverse_loss /= num_steps

    # === Total Loss === #
    total_loss = forward_loss + reverse_loss

    return total_loss, loss_net1
