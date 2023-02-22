#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from masknet import DefocusMaskNet
from DRBNet import DRBNet_single
from dataset import MyCustomDataset

# Define the network architecture
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#net = DRBNet_single().to(device)
net = DefocusMaskNet().to(device)

#define file dir
data_dir = './Cell_gen/v2/mask_input'
gt_dir = './Cell_gen/v2/mask_gt'
#val_dir = './Cell_gen/v2/val_input'
#valgt_dir = './Cell_gen/v2/val_gt'

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Define the training loop
def train(net, dataloader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        #print(inputs.shape, labels.shape)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

# Define the dataset and dataloader
transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = MyCustomDataset(data_dir,gt_dir,transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Train the network for a specified number of epochs
num_epochs = 50
for epoch in range(num_epochs):
    train_loss = train(net, dataloader, criterion, optimizer, device)
    print(f"Epoch {epoch+1} training loss: {train_loss}")
    model_path = "./mask_ckpt/mask_model_e"+str(epoch)+".pkl"
    torch.save(net.state_dict(),model_path)

# Define the testing loop
def test(net, dataloader, criterion, device):
    net.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)



#%%
'''
import numpy as np
import cv2

# Load the input image
img = cv2.imread("input_image.jpg")

# Preprocess the input image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
img = transform(img)
img = img.unsqueeze(0).to(device)

# Predict the defocus mask
net.eval()
with torch.no_grad():
    output = net(img)
mask = output.squeeze().cpu().numpy()

# Visualize the defocus mask
mask = cv2.resize(mask, (img.shape[2], img.shape[3]))
mask = (255*mask).astype(np.uint8)
mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
result = cv2.addWeighted(img.squeeze().permute(1, 2, 0).cpu().numpy(), 0.5, mask, 0.5, 0.0)

# Display the result
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''