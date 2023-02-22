from torch.utils.data.dataset import Dataset
import cv2 as cv
import os
class MyCustomDataset(Dataset):
    def __init__(self, data_dir,gt_dir, transform=None):
         self.data_info = self.get_img_info(data_dir,gt_dir)
         self.transform = transform

    def __getitem__(self, index):
        path_img, path_gt = self.data_info[index]
        img = cv.imread(path_img)
        gt = cv.imread(path_gt)
        if self.transform is not None:
            img = self.transform(img)
            gt = self.transform(gt)
        return img, gt
    
    def __len__(self):
        return len(self.data_info)
    @staticmethod
    def get_img_info(data_dir,gt_dir):
        data_info = list()
        img_names = os.listdir(data_dir)
        for i in range(len(img_names)):
            img_name = img_names[i]
            path_img  = data_dir+'/'+img_name
            path_gt = gt_dir+'/'+img_name
            data_info.append((path_img, path_gt))
        #print(data_info)
        return data_info

class MyCustomDatasetMask(Dataset):
    def __init__(self, data_dir,gt_dir, mask_dir, transform=None):
         self.data_info = self.get_img_info(data_dir,gt_dir,mask_dir)
         self.transform = transform

    def __getitem__(self, index):
        path_img, path_gt, path_mask = self.data_info[index]
        img = cv.imread(path_img)
        gt = cv.imread(path_gt)
        mask = cv.imread(path_mask)
        if self.transform is not None:
            img = self.transform(img)
            gt = self.transform(gt)
            mask = self.transform(mask)
        return img, gt, mask
    
    def __len__(self):
        return len(self.data_info)
    @staticmethod
    def get_img_info(data_dir,gt_dir,mask_dir):
        data_info = list()
        img_names = os.listdir(data_dir)
        for i in range(len(img_names)):
            img_name = img_names[i]
            path_img  = data_dir+'/'+img_name
            path_gt = gt_dir+'/'+img_name
            path_mask = mask_dir+'/'+img_name
            data_info.append((path_img, path_gt,path_mask))
        #print(data_info)
        return data_info