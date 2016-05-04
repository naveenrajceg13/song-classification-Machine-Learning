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
#import essentia
#import essentia.standard as ess
#from extractfeaturefromfile import function_feature
import urllib
import urllib3
import urllib.request
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioFeatureExtraction as aF
import matplotlib 
from matplotlib.pyplot import scatter, xlabel
from sklearn.mixture.gmm import GMM


def build_models(data,class_name,file_name):
   
    print("Building EM")
    #Gausian_EM=GMM(n_components=6,n_init=1000)
    Gausian_EM=GMM(n_components=6)
    Gausian_EM.fit(data)
    print("Building SVM")
    #SVM_Model=SVC(probability=True,kernel='RBF')
    #SVM_Model=SVC(probability=True,kernel='poly')
    SVM_Model=SVC(probability=True)
    print(SVM_Model.fit(data,class_name))
    print("Building Gausian Naive")
    Gausian_Naive_Model=GaussianNB()
    Gausian_Naive_Model.fit(data,class_name)
    print("Building Kmeans")
    K_means_model=KMeans(n_clusters=7)
    K_means_model.fit(data)
    
    return Gausian_EM,SVM_Model,Gausian_Naive_Model,K_means_model