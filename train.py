import os
from torch.utils.data.dataset import Dataset
import numpy as py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import cv2 as cv
import DRBNet
from masknet import DefocusMaskNet 
from dataset import MyCustomDataset, MyCustomDatasetMask
def OverlayL1Loss(img, gt, mask):
    loss = 0.0
    pixels = 0
    size = img.shape
    mask = (mask+1)/2
    #torch.Size([4,3,512,512])
    '''
    for e in range(4):
        for i in range(size[2]):
            for j in range(size[3]):
                if mask[e,:,i,j].mean() == 1:
                    loss += abs(img[e,:,i,j] - gt[e,:,i,j]).mean()
                    pixels += 1
    loss = loss/pixels
    '''
    loss = abs(((img-gt)*mask).mean())
    #print(loss)
    return loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#masknet = DefocusMaskNet().to(device)
#maskckpt = './mask_ckpt/mask_model_e49.pkl'
#masknet.load_state_dict(torch.load(maskckpt))

deblurnet = DRBNet.DRBNet_single().to(device)
data_dir = './Cell_gen/v3/input'
gt_dir = './Cell_gen/v3/gt'
val_dir = './Cell_gen/v3/val_input'
valgt_dir = './Cell_gen/v3/val_gt'
mask_dir = './Cell_gen/v3/mask'

#(0,1)->(-1,1)
norm_mean = (.5,.5,.5)
norm_std = (.5,.5,.5)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std),
])

train_data = MyCustomDatasetMask(data_dir,gt_dir,mask_dir,transform)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=4,shuffle=True)
val_data = MyCustomDataset(val_dir,valgt_dir,transform)
val_loader = torch.utils.data.DataLoader(val_data,batch_size=4,shuffle=True)

optimizer = optim.Adam(deblurnet.parameters(),lr = .0001)
criterion = nn.L1Loss()
for epoch in range(50):
    deblurnet.train()
    runloss = 0.0
    print(epoch)
    for i, data in enumerate(train_loader,0):
        img,gt,mask = data
        img = img.to(device)
        gt = gt.to(device)
        mask = mask.to(device)
        img,gt,mask = Variable(img),Variable(gt),Variable(mask)
        optimizer.zero_grad()
        output = deblurnet(img)
        #mask = masknet(img)
        loss = 0.5*OverlayL1Loss(output,gt,mask)+0.5*criterion(output,gt)
        loss.backward()
        optimizer.step()
        runloss+=loss
    print("Training loss: ",runloss.data/i)
    
    deblurnet.eval()
    valloss=0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader,0):
            img,gt=data
            img = Variable(img.to(device))
            gt = Variable(gt.to(device))
            val = deblurnet(img)
            loss = criterion(val,gt)
            valloss += loss
            '''
            show = (val[0].cpu().numpy()+1)/2
            show = show.resize((512,512,3))
            cv.imshow("o", show)
            cv.waitKey(0)
            cv.destroyAllWindows
            '''
        #print(mask.size())    
        print("Val loss:", valloss.data/i)
        #print(val.shape,  gt.shape)
    model_path = "./deblur_mask_ckpt/cell_model_e"+str(epoch)+".pkl"
    torch.save(deblurnet.state_dict(),model_path)

print('Finish Training')

'''
testimg = torch.reshape(transform(test),(1,3,320,320))
masknet.eval()
with torch.no_grad():
    o = masknet(testimg.to(device))
#deNormalize (-1,1)->(0,1)
o = o/2+0.5
o = np.array(torch.reshape(o.cpu(),(320,320,3)))
#print(o)

cv.imshow('o',o)
cv.waitKey()
cv.destroyAllWindows()
'''
