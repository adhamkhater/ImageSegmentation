from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtWidgets import QMessageBox , QSlider , QLabel
import pyqtgraph as pg
from gui import Ui_MainWindow

import os
import sys
import numpy as np
from PIL import Image
import joblib

import Cmeans
import SNN

from SNN import ShallowNeuralNetworkSegmentation
from sklearn import svm
import SVM


class ApplicationWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        pg.setConfigOption('background', 'w')
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #-----------------------------Constants-----------------------
        self.Cmeans=0
        #----------------Setting up Graphics views--------------------
        self.graphicsViews=[self.ui.graphicsView_1,self.ui.graphicsView_2,
        self.ui.graphicsView_3,self.ui.graphicsView_4,
        self.ui.graphicsView_5,self.ui.graphicsView_6]

        for i in range(6):
            self.graphicsViews[i].getPlotItem().hideAxis('left')
            self.graphicsViews[i].getPlotItem().hideAxis('bottom')
        
        self.inputViews = [self.ui.graphicsView_1,self.ui.graphicsView_3,self.ui.graphicsView_6]

        #------------button and spinner connections--------------------    
        self.ui.Browse.clicked.connect(lambda: self.browse(0))
        self.ui.Browse_2.clicked.connect(lambda: self.browse(1))
        self.ui.Browse_3.clicked.connect(lambda: self.browse(2))

        self.ui.spinBox.valueChanged.connect(self.startCluster)

        self.ui.ApplySNN.clicked.connect(self.ApplySNN)
        self.ui.ApplySVM.clicked.connect(self.ApplySVM)

    def browse(self,n):

            self.filename = QtWidgets.QFileDialog.getOpenFileNames( directory = os.path.dirname(__file__) ,filter= '*.jpeg , *.jpg , *.bmp')
            if((self.filename == ([], '')) | (self.filename ==  0 )):
                return
            print(self.filename[0][0])
            self.extension = os.path.splitext(self.filename[0][0])[1].lower()
            print(self.extension)
            try:
                self.inputViews[n].removeItem(self.myimage)
            except:
                pass
            self.img = Image.open(self.filename[0][0])
            self.myimage=pg.ImageItem(np.asarray(self.img))
            self.myimage.rotate(270)

            self.inputViews[n].clear
            self.inputViews[n].addItem(self.myimage)

    def startCluster(self):
        self.Cmeans=self.ui.spinBox.value()
        if self.Cmeans==(None or 0 or 1):
            return
        result_centroids=Cmeans.main(self.img,self.Cmeans)
        self.drawWindow(result_centroids)

    def drawWindow(self,result_centroids):
        img = Image.new('RGB', (Cmeans.img_width, Cmeans.img_height))
        p = img.load() #RGB pixels of new image
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                RGB_value = result_centroids[Cmeans.getMinDist(Cmeans.px[x, y], result_centroids)] #final assignment of pixel to RGB cluster
                p[x, y] = RGB_value
        #img.show()
        myimage=pg.ImageItem(np.asarray(img))
        myimage.rotate(270)
        self.graphicsViews[1].clear
        self.graphicsViews[1].addItem(myimage)
        #image=ImageOps.grayscale(img)
        #img.show()  
        
    def ApplySNN(self):

        SNN_Model = SNN.loadModel()
        inputImage = SNN.prepareImage(self.img)
        predicted_yvalues = SNN_Model.predict(inputImage)
        outputImage = SNN.imageMeanColoring(predicted_yvalues , inputImage )
        try:
            self.graphicsViews[3].removeItem(self.SNNmyimage)
        except:
            pass
        self.SNNmyimage = pg.ImageItem(np.asarray(outputImage))
        self.SNNmyimage.rotate(270)

        self.graphicsViews[3].clear
        self.graphicsViews[3].addItem(self.SNNmyimage)

    def ApplySVM(self):

        SVM_Model = SVM.loadModel()
        inputImage = SVM.prepareImage(self.img)
        predicted_yvalues = SVM_Model.predict(inputImage)
        outputImage = SVM.imageMeanColoring(predicted_yvalues , inputImage )
        try:
            self.graphicsViews[4].removeItem(self.SVMmyimage)
        except:
            pass
        self.SVMmyimage=pg.ImageItem(np.asarray(outputImage))
        self.SVMmyimage.rotate(270)

        self.graphicsViews[4].clear
        self.graphicsViews[4].addItem(self.SVMmyimage)




def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()