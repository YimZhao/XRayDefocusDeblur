import cv2 as cv
import torch
import DRBNet
import os
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
from dataset import MyCustomDataset

norm_mean = (.5,.5,.5)
norm_std = (.5,.5,.5)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std),
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
testpath = './Cell_gen/v3/val_input'
testgt = './Cell_gen/v3/val_gt'
imgs = os.listdir(testpath)
val_data = MyCustomDataset(testpath,testgt,transform)
val_loader = torch.utils.data.DataLoader(val_data,batch_size=1,shuffle=False)
criterion = nn.L1Loss()

print("*")

for i,data in enumerate(val_loader,0):
    img,gt = data
    img = Variable(img.to(device))
    gt = Variable(gt.to(device))
    with torch.no_grad():
        Net = DRBNet.DRBNet_single().to(device)
        ckptpath = './deblur_ckpt/cell_model_e49.pkl'
        Net.load_state_dict(torch.load(ckptpath))
        output = Net(img)
        loss = criterion(output,gt)
        print(i,loss.cpu().numpy())
    output_cpu = (output.cpu().numpy()[0].transpose(1, 2, 0) +1.0 )/2.0
    input_cpu = (img.cpu().numpy()[0].transpose(1, 2, 0) +1.0 )/2.0
    gt_cpu = (gt.cpu().numpy()[0].transpose(1, 2, 0) +1.0 )/2.0
    cv.imwrite("./result/"+imgs[i],output_cpu*255)
    '''
    cv.imshow("o",output_cpu)
    cv.imshow("i",input_cpu )
    cv.imshow("gt",gt_cpu )
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''