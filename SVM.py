import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
from sklearn import svm
from PIL import Image

import joblib
#Code for training and actual use in GUI will be here, only use the GUI-part of the code however


def startTraining(): # This code must NOT be run, it's put here for showing the code , this handles the training on our training samples



    path = '/trainingSamples/bg'
    path_2 = '/trainingSamples/innercell'
    path_3 = '/trainingSamples/outercell'

    filepaths = [os.path.join(r,file) for r,d,f in os.walk(os.getcwd() + path) for file in f]
    filepaths = [x for x in filepaths if x.endswith(".JPG")]
    filepaths_2 = [os.path.join(r,file) for r,d,f in os.walk(os.getcwd() + path_2) for file in f]
    filepaths_2 = [x for x in filepaths_2  if x.endswith(".JPG")]
    filepaths_3 = [os.path.join(r,file) for r,d,f in os.walk(os.getcwd() + path_3) for file in f]
    filepaths_3 = [x for x in filepaths_3 if x.endswith(".JPG")]

    imglist = [ ]
    imgalgo = [ ]
    Labels = [ ]

    numberOfLabel = 0
    for i in filepaths_2: ## reads inner
        img = Image.open(i)
        newsize = (96, 96)
        img = img.resize(newsize)

        img = np.asarray(img)
        img = img / 255
        for i in range(newsize[0]):
            for j in range(newsize[1]):
                imglist.append(img[i,j])
                Labels.append(numberOfLabel)
    ###############################################
    numberOfLabel = 1
    for i in filepaths_3: ## reads outer
        img = Image.open(i)
        newsize = (96, 96)
        img = img.resize(newsize)

        img = np.asarray(img)
        img = img / 255
        for i in range(newsize[0]):
            for j in range(newsize[1]):
                imglist.append(img[i,j])
                Labels.append(numberOfLabel)
    ##################################################
    numberOfLabel = 2
    for i in filepaths: ## reads back ground
        img = Image.open(i)
        newsize = (96, 96)
        img = img.resize(newsize)

        img = np.asarray(img)
        img = img / 255
        for i in range(newsize[0]):
            for j in range(newsize[1]):
                imglist.append(img[i,j])
                Labels.append(numberOfLabel)
    ####################################################

    X = np.asarray(imglist)
    Y = np.asarray(Labels)

    clf = svm.SVC( C=100 , kernel='linear' , gamma="scale" )
    clf.fit(X,Y)

    joblib_file = "SVM_Model_1_test.pkl"
    joblib.dump(clf, joblib_file)
    
#GUI-related code
def prepareImage(image):

    newimg_1 = [ ]
    #newsize = (128, 128)
    #img_1 = image.resize(newsize)
    img_1 = image
    img_1 = np.array(img_1)
    img_1 = img_1.astype("float64")
    img_1 = img_1 / 255
    global w 
    global h 
    w = image.width
    h = image.height
    for i in range(w):
        for j in range(h):
            newimg_1.append(img_1[i,j])
    newimg_1 = np.array(newimg_1)
    return newimg_1

def imageMeanColoring(y_pred,testimg): # testimg must be a numpyarray

    cluster_mean_inner  = []
    cluster_mean_outer = []
    cluster_mean_bg = []

    for i in range(len(testimg)):
        if y_pred[i] == 0:
            cluster_mean_inner.append(testimg[i])
        if y_pred[i] == 1:
            cluster_mean_outer.append(testimg[i])
        if y_pred[i] == 2:
            cluster_mean_bg.append(testimg[i])


    # Coloring preparation , get means of the pixels
    labelcount = 0
    img_inner_mean = np.mean(cluster_mean_inner, axis=(0))
    img_outer_mean = np.mean(cluster_mean_outer, axis=(0))
    img_background_mean = np.mean(cluster_mean_bg, axis=(0))

    newImage = np.zeros((w, h, 3), dtype=np.uint8)

    for i in range(w):
        
        for j in range(h): # Manually replace each pixel with the new color
            
            if y_pred[labelcount] == 0:
                newImage[i,j] = img_inner_mean*255
            elif y_pred[labelcount] == 1:
                newImage[i,j] = img_outer_mean*255
            else:
                newImage[i,j] = img_background_mean*255
                
            labelcount += 1
            
    newImageConverted = Image.fromarray(newImage)  # Convert it into image object for preview
    return newImageConverted

def loadModel():
    joblib_file = "SVM_Model_1.pkl"
    joblib_model = joblib.load(joblib_file)

    return joblib_model