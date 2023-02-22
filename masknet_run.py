import numpy as np
import cv2
import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from masknet import DefocusMaskNet
# Load the input image
img = cv2.imread("input_image.jpg")

# Preprocess the input image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img = transform(img)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img = img.unsqueeze(0).to(device)

# Predict the defocus mask
net = DefocusMaskNet()
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
