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

#import My modules
from Process_Images_HDA import Image_extraction, Patches_extraction, Label_patches, preprocess
from Visualize_Images_HDA import Visualization, Patch_Visualization
from First_NN_HDA import CNN, Compile_Train_cnn, Model_history, Model_save, Load_Model, Model_evaluation, Image_prediction
from Second_NN_HDA import CNN_2, Compile_Train_cnn_2, Model_history_2, Model_save_2, Load_Model_2, Model_evaluation_2, Image_prediction_2
from Third_NN_HDA import CNN_3, Compile_Train_cnn_3, Model_history_3, Model_save_3, Load_Model_3, Model_evaluation_3, Image_prediction_3
from Fourth_NN_HDA import CNN_4, Compile_Train_cnn_4, Model_history_4, Model_save_4, Load_Model_4, Model_evaluation_4, Image_prediction_4

def main():
    #From Here we can Call the methods that we need in a specific Moment
main()
