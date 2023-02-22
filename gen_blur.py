from prop_class_asm import propagate
import cv2 
import numpy as np
import os
import random

imgpath = "E:/Thesis/BNL/Deblurring/MyCode/data/Cell/"
imgsavepath = "E:/Thesis/BNL/Deblurring/MyCode/Cell_gen/v2/input/"
gtsavepath = "E:/Thesis/BNL/Deblurring/MyCode/Cell_gen/v2/gt/"
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
        '''
        cv2.imshow("pro",bg_p/255)
        cv2.waitKey(0)
        cv2.destroyAllWindows
        '''
        #cv2.imshow("bg",bggray) x 
       
        result= np.zeros(imgsize)
        gt = np.zeros(imgsize)
        result = bg_p+imggray
        gt = bggray+imggray
        for i in range(imgsize[0]):
            for j in range(imgsize[1]):
                if result[i,j] == 0:
                    result[i,j] = 10
                if result[i,j] > 255:
                    result[i,j] = 255
                if gt[i,j] == 0:
                    gt[i,j] = 10
                if gt[i,j] > 255:
                    gt[i,j] = 255
        #print(result)
        '''
        cv2.imshow("ori",imggray)
        cv2.imshow("prop",result/255)
        cv2.imshow("mask", gt)
        cv2.waitKey(0)
        cv2.destroyAllWindows
        '''
        cv2.imwrite(imgsavepath+file[0:-4]+bgfile[0:-4]+".jpg",result)
        cv2.imwrite(gtsavepath+file[0:-4]+bgfile[0:-4]+".jpg",gt)
       
