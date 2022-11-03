# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:33:38 2022

@author: Ibrahim
"""


import numpy as np
import cv2
import os.path


Dataset_path = r"D:\MS courses\Third Semester\Deep learning\Project\neural nets\pets"
Prediction_files_path = r"D:\MS courses\Third Semester\Deep learning\Project\neural nets\pets\dogs"
labels = ["Duck","Horse"]




def pre_processing(image_path):
    '''
    Images are first converted from RGB to grey and then the images are reduced in size. Then these reducded images
    are normalized by dividing them with 255. 
    
    Input:
    - image_path: This is the path of the image that is to be pre_processed.
    
    Returns:
    - img_pred: The processed image, it is first converted into grey and then resized to 100x100 and then normalized.
    
    '''
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pred = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    
    img_pred = np.asarray(img_pred)
    
    img_pred = img_pred / 255
    return img_pred



def train(path):
    X = []
    for file in os.listdir(path):
            if (os.path.isfile(path + "/" + file)):
                image = pre_processing(path + "/" + file)
                image = np.reshape(image,(image.shape[0]*image.shape[1]))
                X.append(image)
    
    X_train = np.array(X)   # Shape is (20,10000) (r,c) 
    
    y_duck =  np.zeros((10,1))
    y_horse = np.ones((10,1))
    Y_train = np.concatenate((y_duck,y_horse))     # Shape is (20,1) (r,c)
    
    return X_train,Y_train

def test(path):
    X = []
    for file in os.listdir(path):
            if (os.path.isfile(path + "/" + file)):
                image = pre_processing(path + "/" + file)
                image = np.reshape(image,(image.shape[0]*image.shape[1]))
                X.append(image)
    
    X_test = np.array(X)   # Shape is (10,10000) (r,c) 
    
    y_duck =  np.zeros((5,1))
    y_horse = np.ones((5,1))
    Y_test = np.concatenate((y_duck,y_horse))     # Shape is (10,1) (r,c)
    
    return X_test,Y_test


X_train,Y_train = train(Dataset_path)
print(X_train.shape)
print(Y_train.shape)
            