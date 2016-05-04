from extra_from_file import extract_fur
from BuildModels import build_models
from sklearn import preprocessing, neural_network
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans,affinity_propagation,estimate_bandwidth
from sklearn.neural_network import BernoulliRBM
import sklearn.neural_network
from sklearn.svm import SVC
import numpy as np
import csv
import sklearn.svm
from sklearn.naive_bayes import BaseNB
import urllib
import urllib3
import urllib.request
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioFeatureExtraction as aF
import matplotlib 
from matplotlib.pyplot import scatter, xlabel
from sklearn.mixture.gmm import GMM
from pyAudioAnalysis import audioBasicIO
import numpy
import os
import pydub
from converttofrequency import count
import sys
from PyQt4 import QtGui,QtCore
from msilib.schema import CheckBox
from Cython.Plex.Lexicons import State
import pyglet
import os
from pydub import AudioSegment



data=[[]]
class_name=[]
filename=[]
Gausian_EM_model=" "
SVM_Model="" 
Gausian_Naive_Mode=""
K_means_model=""
cluster_em=""
GUI=""
selected_file=""
class Window(QtGui.QMainWindow):
    def __init__(self,filename):
        super(Window,self).__init__()
        self.setGeometry(50,50,1000,600)
        self.setWindowTitle("Song Classification")
        self.setWindowIcon(QtGui.QIcon('D:\python.png'))
        
        extractAction=QtGui.QAction("&Get to choppa",self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip('Leave the app')
        extractAction.triggered.connect(self.close_application)
        self.statusBar()
        mainMenu=self.menuBar()
        fileMenu=mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction)
        self.filename=filename
        self.addfiles()
        self.selected_file_value=""
        
    def close_application(self):
        choice=QtGui.QMessageBox.question(self,'Extract!',
                                          "get into the exit?",QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
        if choice==QtGui.QMessageBox.Yes:
            print("Exit naaaaaaaaaaaaa")
            sys.exit()
        else:
            pass
    
    def addfiles(self):
        
        self.styleChoice=QtGui.QLabel("Select a song",self)
        self.comboBox=QtGui.QComboBox(self)
        self.comboBox.setMinimumWidth(400)
        self.comboBox.setSizeAdjustPolicy(True)
        self.comboBox.move(450,200)
        self.comboBox.activated[str].connect(self.style_choice)
        self.styleChoice.move(350,200)
        self.styleChoice1=QtGui.QLabel("Selected File:",self)
        self.styleChoice1.setMinimumWidth(700)
        self.styleChoice1.move(150,230)
        self.styleChoice2=QtGui.QLabel("",self)
        self.styleChoice2.setMinimumWidth(500)
        self.styleChoice2.setMinimumHeight(300)
        self.styleChoice2.move(15,290)
        
        openfile = QtGui.QAction('&Open File',self)
        openfile.setShortcut("Ctrl+F+F")
        openfile.setStatusTip('choose file')
        openfile.triggered.connect(self.file_open)
        
        self.styleChoice3=QtGui.QLabel("Select a song to test the class",self)
        self.styleChoice3.setMinimumWidth(200)
        self.styleChoice3.move(250,100)
        openfile=QtGui.QPushButton("Open File",self)
        openfile.clicked.connect(self.file_open)
        openfile.resize(70,20)
        openfile.move(450,100)
        
        getclass=QtGui.QPushButton("get class",self)
        getclass.clicked.connect(self.open_svm)
        getclass.resize(100,30)
        getclass.move(550,550)
        
        
        getclass1=QtGui.QPushButton("get Probability",self)
        getclass1.clicked.connect(self.open_EM)
        getclass1.resize(100,30)
        getclass1.move(350,550)
        #getclass1.minimumWidth(200)
        self.styleChoice34=QtGui.QLabel("Select a cluster",self)
        self.comboBox1=QtGui.QComboBox(self)
        self.comboBox1.setMinimumWidth(90)
        self.comboBox1.setSizeAdjustPolicy(True)
        self.comboBox1.addItem("1")
        self.comboBox1.addItem("2")
        self.comboBox1.addItem("3")
        self.comboBox1.addItem("4")
        self.comboBox1.addItem("5")
        self.comboBox1.addItem("6")
        self.comboBox1.addItem("7")
        self.styleChoice34.move(650,550)
        self.comboBox1.move(750,550)
        self.comboBox1.activated[str].connect(self.style_cluster)

        for each in self.filename:
            self.comboBox.addItem(str(each))
        self.show()
        print("completed")
    
    def open_EM(self):
        getclass_EM()
    def open_svm(self):
        getclass_svm() 
        
    def file_open(self):
        name=QtGui.QFileDialog.getOpenFileName(self,'Open File')
        self.selected_file_value=name
        self.styleChoice1.setText("Selected File:  "+name)
        check_class(name)   
       
    def style_cluster(self,text):
        getcluster(text)    
    def style_choice(self,text):
        
        self.styleChoice1.setText("Selected File:  "+text)
        self.show()
        self.selected_file_value=text
        check_function(text)
        #QtGui.QApplication.setStyle(QtGui.QStyleFactory.create(text))
    def set_recommendation(self,text):
        self.styleChoice2.setText("Recommendations \n"+text)
    
    def set_cluster(self,text):
        self.styleChoice2.setText(text)
        
    def showpopup(self,text):
        choice=QtGui.QMessageBox.question(self,'Class of Song',
                                          text,QtGui.QMessageBox.Ok)


def getcluster(location):
    values="files in cluster "+location+" are\n"
    values=values+getfilesincluster(clusters_k[int(location)-1], filename)
    GUI.set_cluster(values)

def getclass_svm():
    classes=['Country','EDM','Jazz','Rap','Rock','punk']
    print("selected file",GUI.selected_file_value)
    value=predictmp3class(GUI.selected_file_value,SVM_Model)
    for each in value:
        value=each
    if value==7:
        value=6
    GUI.showpopup("Class value of the song"+GUI.selected_file_value+" is "+classes[value-1])
    
def getclass_EM():
    classes=['Country','EDM','Jazz','Rap','Rock','punk']
    print("selected file",GUI.selected_file_value)
    value=getfileprobability(GUI.selected_file_value,Gausian_EM_model)
    str1=""
    count=0
    for each in value:
        for each1 in each:
            str1=str1+"class "+str(count+1)+" is "+str(round(each[count]*100))+"\n"
            count=count+1
        break
    GUI.showpopup("Probality value of the song \n"+GUI.selected_file_value+" is "+str1)
    

def check_class(filename_g):
    selected_file=filename_g
    #os.system("start "+filename_g)
    print(selected_file)    
        
def check_function(filename_g):
    selected_file=filename_g
    index=filename.index(filename_g)
    recomm=getnear(cluster_em,index)
    string_value=""
    count=0
    print(index,recomm)
    
    for each in recomm:
        string_value=string_value+str(count+1)+") "+filename[int(each)]+"\n"
        count=count+1
        if count==10:
            break
    GUI.set_recommendation(string_value)
    #os.system("start "+'r'+filename_g)
    print(string_value)
    string_value=""
def getfeaturedetailsfromfile():
    [features, fileNames]=(aF.dirWavFeatureExtraction("C:\project\songs\A", 1.0, 1.0, aT.shortTermWindow,aT.shortTermStep,True))
    return features
def deletefilefromdirec(dirname):
    folder = dirname
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
def convertowave(filename):
    sound = pydub.AudioSegment.from_mp3(filename)
    sound = sound.set_channels(1)
    sound.export("C:\\project\\songs\\A\\A.wav", format="wav") 
    
def predictmp3class(filename,SVM_Model):
    #convertowave("C:\\project\\songs\\music\\A\\04. I Don't Care.mp3")
    convertowave(filename)
    feature_input=getfeaturedetailsfromfile()
    deletefilefromdirec("C:\project\songs\A")
    return SVM_Model.predict(feature_input)

def getclusters(data,K_means_model):
    
    clusters=[[]]
    for i in "123456":
        temp=[]
        clusters.append(temp)
    count=0
    for each in data:
        value=K_means_model.predict(data[count])
        for each in value:
            value=each
        clusters[int(value)].append(count)
        count=count+1
    return clusters
def getfilesincluster(cluster_row,filename):
    
    str1=""
    for each in cluster_row:
        str1=str1+filename[each]+"\n"
    return str1

def getfileprobability(filename,Gausian_Naive_Model):
    convertowave(filename)
    feature_input=getfeaturedetailsfromfile()
    values=Gausian_Naive_Model.predict_proba(feature_input)
    count=0
    deletefilefromdirec("C:\project\songs\A")
    return values

def formcluterprobability(data,em_model):
    cluster_e=[[]]
    count=0
    for each in data:
        if count!=0:
            temp=[]
            cluster_e.append(temp)
        values=em_model.predict_proba(data[count])
        acount=0
        for each in values:
            for each1 in each:
                cluster_e[count].insert(acount,(round(each[acount]*100)))
                acount=acount+1
            break
        count=count+1
    return cluster_e

def getnear(cluster_v,data_value):
    max=0
    index=0
    count=0
    for each in cluster_v[data_value]:
        if(max<float(each)):
            max=float(each)
            index=count
        count=count+1
    second_max=0
    second_index=0
    count=0
    for each in cluster_v[data_value]:
        if(count==index):
            continue
        if(second_max<float(each)):
            second_max=float(each)
            second_index=count
        count=count+1
    reco=[]
    count=0
    if(second_index==0):
        second_index=index
    #print(cluster_v[data_value][index],cluster_v[data_value][second_index])
    for each in cluster_v:
        if(count==data_value):
            count=count+1
            continue
        flag_check=True
        #number >= 10000 and number <= 30000:
        #print(cluster_v[count][index],cluster_v[count][second_index])
        if((cluster_v[data_value][index]>cluster_v[count][index])):
            if((cluster_v[data_value][index]<=cluster_v[count][index]+5)):
                reco.append(count)
                count=count+1
                continue
        if((cluster_v[data_value][second_index]>cluster_v[count][second_index])):
            if((cluster_v[data_value][second_index]<=cluster_v[count][second_index]+2)):
                reco.append(count)
                count=count+1
                continue
        if((cluster_v[data_value][index]<cluster_v[count][index]-5)):
            if((cluster_v[data_value][index]>=cluster_v[count][index])):
                reco.append(count)
                count=count+1
                continue
        if((cluster_v[data_value][second_index]<cluster_v[count][second_index]-2)):
            if((cluster_v[data_value][second_index]>=cluster_v[count][second_index])):
                reco.append(count)
                count=count+1
                continue
            
        count=count+1
    #print(reco)
    return reco

data,class_name,filename=extract_fur()
#print(len(data))
#print(len(class_name))
#print(len(filename))
Gausian_EM_model,SVM_Model,Gausian_Naive_Model,K_means_model=build_models(data,class_name,filename)
predictmp3class("C:\\project\\songs\\music\\A\\04. I Don't Care.mp3",SVM_Model)
clusters_k=getclusters(data,K_means_model)
cluster_em=formcluterprobability(data,Gausian_EM_model)
#print(getnear(cluster_em,366))
app=QtGui.QApplication(sys.argv)
GUI=Window(filename)
sys.exit(app.exec_())
#GUI.addfiles()

#listallfiles(filename)

