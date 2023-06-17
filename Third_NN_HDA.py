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

def CNN_3():
    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', input_shape=[32, 32, 3]))
    cnn.add(BatchNormalization())
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(3, activation='softmax'))

    cnn.summary()
    
    return cnn

def Compile_Train_cnn_3(cnn,X_train, Y_train,X_test, Y_test):
    batch_size=128
    nb_epochs=100
    cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    My_keras_model=cnn.fit(X_train, Y_train, batch_size = batch_size, epochs = nb_epochs, verbose = 1, validation_data = (X_test, Y_test))
    
    return My_keras_model

def Model_history_3(My_keras_model):
    
    print(My_keras_model.history.keys())

    plt.plot(My_keras_model.history['accuracy'])
    plt.plot(My_keras_model.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(My_keras_model.history['loss'])
    plt.plot(My_keras_model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def Model_save_3(cnn):
    cnn.save("Lymphoma_CNN_Classifier_third_model.h5")
    
def Load_Model_3():
    from keras.models import load_model
    new_model = load_model("Lymphoma_CNN_Classifier_third_model.h5")
    
    return new_model
    
def Model_evaluation_3(X_test, Y_test):
    score = Load_Model_2().evaluate(X_test, Y_test, verbose = 0) 

    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])
    
def Image_prediction_3(path="tests_predictions/cll.tif"):
    
    X=cv2.imread(path)
    X=X/255
    patch_height=32
    patch_width =32
    max_patch=0.01

    X_1=feature_extraction.image.extract_patches_2d(X,(patch_height, patch_width),max_patches=max_patch)
    values=Load_Model_2().predict(X_1)

    total=np.zeros(3)
    for el in values:
        if el[0]>el[1] and el[0]>el[2]:
            total[0]=total[0]+1
        elif el[1]>el[0] and el[1]>el[2]:
            total[1]=total[1]+1
        elif el[2]>el[0] and el[2]>el[1]:
            total[2]=total[2]+1
    print(total)
    if el[0]>el[1] and el[0]>el[2]: print("Prediction: CLL")
    elif el[1]>el[0] and el[1]>el[2]: print("Prediction: FL")
    elif el[2]>el[1] and el[2]>el[0]: print("Prediction: MCL")
