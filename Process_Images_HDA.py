#Install dependencies
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from time import time
import pandas as pd
import tensorflow as tf
import sklearn
import random
import pathlib
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
import keras
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.decomposition import PCA


def Image_extraction(): #OK 
    #In this section I will import images into a an array made by matrices
    #It will be evaluated also time

    DATADIR = "data"

    #!wget https://drive.google.com/file/d/1VRRptGC_1OSRaA_WWl28X0M_Fhtfzjby/view?usp=sharing

    CATEGORIES = ["CLL", "FL","MCL"] #Nota: CLL=0 , FL=1 , MCL=2 in labels

    Cat_array=[]
    img_array=[]
    y_array=[]
    counter=0
    NoneType =type(None)
    Time1=time()

    classification=-1

    for category in CATEGORIES:  
        path = os.path.join(DATADIR,category) 
        
        classification=classification+1 
        Cat_array.append(counter)
        
        print(path+str(counter))
        
        for img in os.listdir(path): 
            img_array.append(cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR))
            
            #Se non si opera in questo modo l'immagine 141 risulterà tipizzata come NoneType
            if isinstance(img_array[counter], NoneType):
                img_array.pop(counter)
            else: 
                y_array.append(classification)
                counter=counter+1

    y_array=np.array(y_array)
    
    print("Tempo di importazione immagini "+str(time()-Time1)) 
    print("Immagini importate: "+str(counter))
    print(Cat_array)
    
    return img_array, Cat_array

def Patches_extraction(img_array):
    #Extract Patches from Images
    
    patch_height=32
    patch_width =32
    max_patch=0.00015 #0.0015

    Array_patches=[]
    counter=0

    Time1=time()
    counter=0
    for image in img_array:
        image=np.array(image)
        Array_patches.append(feature_extraction.image.extract_patches_2d(image,(patch_height, patch_width),max_patches=max_patch))
        counter=counter+1
    print("Tempo di estrazione patches "+str(time()-Time1))

    return Array_patches

def Label_patches(Array_patches, Cat_array):
    #Calcola numero di patch totali e crea una nuova label per l'array di patches
    Total_number=0
    counter=0
    y_patch=[]
    for el in Array_patches:
        if counter<Cat_array[1]:
            Cat=0
        elif counter<Cat_array[2]:
            Cat=1
        else:
            Cat=2
            
        Total_number=Total_number+len(el)
        
        for i in range(len(el)):
            y_patch.append(Cat)

        counter=counter+1
        
    print(Total_number)

    return y_patch

def check(Y_test):
    a, b, c=0, 0, 0
    for el in Y_test: 
        if el[0]==1:
            a=a+1
        elif el[1]==1:
            b=b+1
        elif el[2]==1:
            c=c+1
            
    print(a,b,c)

    
def preprocess(Array_patches, y_patch):
    patches=[]#"Vec form" of our patch matrix
    for el in Array_patches:
        for a in el:
            patches.append(a)
            
    patches=np.array(patches)
    y_patch=np.array(y_patch)

    m=len(patches)
    m_training = int(0.8*m)
    m_test =  int(0.2*m)

    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig=train_test_split(patches, y_patch, test_size=m_test, train_size=m_training)

    print ("number of training examples = " + str(X_train_orig.shape[0]))
    print ("number of test examples = " + str(X_test_orig.shape[0]))
    print ("X_train shape: " + str(X_train_orig.shape))
    print ("Y_train shape: " + str(Y_train_orig.shape))
    print ("X_test shape: " + str(X_test_orig.shape))
    print ("Y_test shape: " + str(Y_test_orig.shape))


    #Rescaling pixels from range 0-255 to their normalized version
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    Y_train = to_categorical(Y_train_orig)
    Y_test = to_categorical(Y_test_orig)

    check(Y_test)

    Y_train=np.array(Y_train)
    Y_test=np.array(Y_test)
    
    return X_train, Y_train, X_test, Y_test
