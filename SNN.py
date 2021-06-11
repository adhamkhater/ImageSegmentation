import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score

from skimage.feature import hog
import pandas as pd

from scipy.special import expit, logit
import joblib

#Code for training and actual use in GUI will be here, only use the GUI-part of the code however

class ShallowNeuralNetworkSegmentation:  # The actual class of the Shallow Neural Network, handles like a normal model
    def __init__(self, epochs = 20, hiddenLayerNodes = 3 , learningRate = 1):
        
        self.Epochs = epochs
        self.n_h = hiddenLayerNodes
        self.learning_rate = learningRate
        
        
    def sigmoid(self,z):
        
        return expit(z)

    def softmax(self,z):
        
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)
    
    def fit(self, x , y):
        
        self.X_train = x
        self.y_train = y
        
        
        n_x = x.shape[1]
        n_y = y.shape[1]

        # initialization
        
        self.params = { 
                   "W1": np.random.randn(n_x, self.n_h ) ,
                   "b1": np.zeros((1, self.n_h )) ,
                   "W2": np.random.randn(self.n_h , n_y) ,
                   "b2": np.zeros((1, n_y)) 
                        }
        
        self.cache = {
                   "A1": None ,
                   "A2": None ,
                   "Z1": None ,
                   "Z2": None ,
                        }
        
        self.grads = { 
                   "dW1": None ,
                   "dW2": None ,
                        }
        
        for i in range(self.Epochs):
            self.feed_forward()
            self.back_propagate()
            self.update()
            print("Epoch:{} done".format(i+1))
            
    def back_propagate(self):

        dW2 = (self.cache["A2"] - self.y_train) / (self.y_train.shape[0])
        
        dZ1 = np.dot(dW2 , self.params["W2"].T)
        dW1 = dZ1 * ( self.cache["A1"] * (1 - self.cache["A1"]) )

        self.grads = {"dW1": dW1,"dW2": dW2}

        
    def feed_forward(self):

        self.cache["Z1"] = np.dot( self.X_train , self.params["W1"]) + self.params["b1"]

        self.cache["A1"] = self.sigmoid( self.cache["Z1"])

        self.cache["Z2"] = np.dot( self.cache["A1"] , self.params["W2"] ) + self.params["b2"]

        self.cache["A2"] = self.softmax( self.cache["Z2"])

    def update(self):
        
        self.params["W1"] = self.params["W1"] - self.learning_rate * np.dot(self.X_train.T, self.grads["dW1"] )
        self.params["W2"] = self.params["W2"] - self.learning_rate * np.dot(self.cache["A1"].T, self.grads["dW2"] )

        self.params["b1"] = self.params["b1"] - self.learning_rate * np.sum(self.grads["dW1"] , axis=0)
        self.params["b2"] = self.params["b2"] - self.learning_rate * np.sum(self.grads["dW2"] , axis=0, keepdims=True)
   
    def predict(self, x_test):
        
        y_pred = [ ]
        for i in x_test:
            self.X_train = i
            self.feed_forward()
            y_pred.append(self.cache["A2"].argmax())
        return y_pred
    
    def returnCache(self):
        return self.cache

def startTraining(): # This code must NOT be run, it's put here for showing the code , this handles the training on our training samples
    path = '/Segmentation_samples'

    filepaths = [os.path.join(r,file) for r,d,f in os.walk(os.getcwd() + path) for file in f]
    filepaths = [x for x in filepaths if x.endswith(".PNG")]

    imglist = [ ]
    Labels = [ ]

    numberOfLabel = 0
    for i in filepaths:
        img = Image.open(i)
        newsize = (96, 96)
        img = img.resize(newsize,Image.LANCZOS)

        img = np.asarray(img)
        img = np.delete(img,3,axis=2)  # The sample data was in PNG (and read in 4 dimensions, due to a bug in snipping) and final column had to be removed
        img = img.astype("float64")
        img = img / 255
        for i in range(newsize[0]):
            for j in range(newsize[1]):
                imglist.append(img[i,j])
                Labels.append(numberOfLabel)
        numberOfLabel += 1
        if numberOfLabel == 3:
            numberOfLabel = 0

    X_train = np.asarray(imglist)

    #Binary encoding
    s = pd.Series(Labels)
    d = pd.get_dummies(s)
    Y_train = d.to_numpy()

    epochs = 100
    hiddenLayerNodes = 3
    learningRate = 4
    SNN_model = ShallowNeuralNetworkSegmentation(epochs , hiddenLayerNodes, learningRate)
    SNN_model.fit(X_train,Y_train)

    joblib_file = "SNN_Model_1_test.pkl"
    joblib.dump(SNN_model, joblib_file)

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
    joblib_file = "SNN_joblib_model4.pkl"
    joblib_model = joblib.load(joblib_file)

    return joblib_model