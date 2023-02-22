from prop_class_asm import propagate
import cv2 
import numpy as np
import os
import random

imgpath = "E:/Thesis/BNL/Deblurring/MyCode/data/Cell/"
imgsavepath = "E:/Thesis/BNL/Deblurring/MyCode/Cell_gen/v3/input/"
gtsavepath = "E:/Thesis/BNL/Deblurring/MyCode/Cell_gen/v3/gt/"
masksavepath = "E:/Thesis/BNL/Deblurring/MyCode/Cell_gen/v3/mask/"
imgfiles = os.listdir(imgpath)
bgfiles = random.sample(imgfiles,len(imgfiles))

#print(bgfiles)
for file, bgfile in zip(imgfiles,bgfiles):
    if not os.path.isdir(file):
        imgsize = (512,512)
        img = cv2.imread(imgpath+"/"+file)
        img = cv2.resize(img,imgsize)
        bg = cv2.imread(imgpath+'/'+ bgfile)
        bg = cv2.resize(bg,imgsize)
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bggray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        bg_p = np.real(propagate(bggray,1,1,1,1,10))
        for i in range(imgsize[0]):
            for j in range(imgsize[1]):
                if bg_p[i,j] < 0:
                    bg_p[i,j] = 0
                if bg_p[i,j] > 255:
                    bg_p[i,j] = 255
        #bg_p = bg_p.astype(np.uint8)
        '''
        cv2.imshow("pro",bg_p/255)
        cv2.waitKey(0)
        cv2.destroyAllWindows
        '''
        #cv2.imshow("bg",bggray) x 
       
        result= np.zeros(imgsize)
        gt = np.zeros(imgsize)
        #overlay mask
        mask = np.zeros(imgsize)
        result = bg_p.astype(np.uint8)+imggray
        gt = bggray+imggray

        for i in range(imgsize[0]):
            for j in range(imgsize[1]):
                if result[i,j] > 255:
                    result[i,j] = 255

                if bg_p[i,j]>10 and imggray[i,j]>10:
                    mask[i,j] = 255
                else:
                    mask[i,j] = 0

                if gt[i,j] > 255:
                    gt[i,j] = 255
                
        #print(result)
        '''
        cv2.imshow("ori",imggray)
        cv2.imshow("prop",result)
        cv2.imshow("gt", gt)
        cv2.imshow("mask",mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows
        print(gt.max())
        '''
        cv2.imwrite(imgsavepath+file[0:-4]+bgfile[0:-4]+".png",result.astype(np.uint8))
        cv2.imwrite(gtsavepath+file[0:-4]+bgfile[0:-4]+".png",gt.astype(np.uint8))
        cv2.imwrite(masksavepath+file[0:-4]+bgfile[0:-4]+".png",mask.astype(np.uint8))
