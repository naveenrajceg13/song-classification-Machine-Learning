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
import matplotlib.pyplot as plt
#from sklearn.neural_network import MLPClassifier

def featuretoarray(data,features,class_value,first_time,count,class_name1,file_name,fileNames):
    
    acount=0
    for each in features:
        if first_time!=True:
            temp=[]
            data.append(temp)
        for each1 in each:
            data[count].append(each1)
        count=count+1
        class_name1.append(class_value)
        first_time=False
        file_name.append(fileNames[acount])
        acount=acount+1
        
    return data,class_name1,first_time,count,file_name
def featuretoarray1(data,features,class_value,first_time,count,class_name1):
    
    ocount=0
    acount=0
    print("features length",len(features))
    for each in features:
        if first_time!=True:
            temp=[]
            data.append(temp)
        for each1 in each:
            print("each length",len(each))
            if acount>=len(each):
                break
            for each2 in each1:
                data[count].append(each2)
                ocount=ocount+1
                print("each 1 len",len(each1))
                if ocount>=len(each1):
                    break
            ocount=0
            acount=acount+1
            if acount>=len(each):
                break
            first_time=False
        acount=0
        class_name1.append(class_value)
        count=count+1
    return data,class_name1,first_time,count
#aT.featureAndTrain(["A","B"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
#print(aF.dirsWavFeatureExtraction("A", 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, False))
#aF.mtFeatureExtractionToFileDir("A", 1.0, 1.0, aT.shortTermWindow,aT.shortTermStep, True, True, True)
data=[[]]
class_name1=[]
file_name=[]
print("first")
#print(aF.mtFeatureExtractionToFileDir("A", 1.0, 1.0, aT.shortTermWindow,aT.shortTermStep, True, True, True))
print("extracting feature -1")
[features, fileNames]=(aF.dirWavFeatureExtraction("C:\project\songs\music_wave\B", 1.0, 1.0, aT.shortTermWindow,aT.shortTermStep,True))

print("extracting feature -2")
[features1, fileNames1]=(aF.dirWavFeatureExtraction("C:\project\songs\music_wave\B", 1.0, 1.0, aT.shortTermWindow,aT.shortTermStep,True))
print("extracting feature -3")
[features2, fileNames2]=(aF.dirWavFeatureExtraction("C:\project\songs\music_wave\C", 1.0, 1.0, aT.shortTermWindow,aT.shortTermStep,True))
print("extracting feature -4")
[features3, fileNames3]=(aF.dirWavFeatureExtraction("C:\project\songs\music_wave\D", 1.0, 1.0, aT.shortTermWindow,aT.shortTermStep,True))
print("extracting feature -5")
[features4, fileNames4]=(aF.dirWavFeatureExtraction("C:\project\songs\music_wave\E", 1.0, 1.0, aT.shortTermWindow,aT.shortTermStep,True))
print("extracting feature -6")
#[features5, fileNames5]=(aF.dirWavFeatureExtraction("C:\project\songs\music_wave\F", 1.0, 1.0, aT.shortTermWindow,aT.shortTermStep,True))
print("extracting feature -7")
[features6, fileNames6]=(aF.dirWavFeatureExtraction("C:\project\songs\music_wave\G", 1.0, 1.0, aT.shortTermWindow,aT.shortTermStep,True))
#[features, classNames, fileNames]=aF.dirsWavFeatureExtraction("A", 1.0, 1.0, aT.shortTermWindow, aT.shortTermWindow, True)
print("second") 
#[features1, classNames1, fileNames1]=aF.dirsWavFeatureExtraction("C:\project\songs\B", 1.0, 1.0, aT.shortTermWindow, aT.shortTermWindow, True)
#print("third")
#[features2, classNames2, fileNames2]=aF.dirsWavFeatureExtraction("C:\project\songs\C", 1.0, 1.0, aT.shortTermWindow, aT.shortTermWindow, True)
print("forming matrix -1")

data,class_name1,first_time,count,file_name=featuretoarray(data,features, 1, True,0,class_name1,file_name,fileNames)
print("size of data",len(data))
print("size of class",len(class_name1))
print("size of file name",len(file_name))

print("forming matrix -2")
data,class_name1,first_time,count,file_name=featuretoarray(data,features1, 2, False,count,class_name1,file_name,fileNames1)
print("size of data",len(data))
print("size of class",len(class_name1))
print("size of file name",len(file_name))
print("forming matrix -3")
data,class_name1,first_time,count,file_name=featuretoarray(data,features2, 3, False,count,class_name1,file_name,fileNames2)
print("size of data",len(data))
print("size of class",len(class_name1))
print("size of file name",len(file_name))
print("forming matrix -4")
data,class_name1,first_time,count,file_name=featuretoarray(data,features3, 4, False,count,class_name1,file_name,fileNames3)
print("size of data",len(data))
print("size of class",len(class_name1))
print("size of file name",len(file_name))
print("forming matrix -5")
data,class_name1,first_time,count,file_name=featuretoarray(data,features4, 5, False,count,class_name1,file_name,fileNames4)
print("size of data",len(data))
print("size of class",len(class_name1))
print("size of file name",len(file_name))
print("forming matrix -6")
#data,class_name1,first_time,count=featuretoarray(data,features5, 6, False,count,class_name1,file_name,fileNames5)
print("forming matrix -7")
data,class_name1,first_time,count,file_name=featuretoarray(data,features6, 7, False,count,class_name1,file_name,fileNames6)
print("size of data",len(data))
print("size of class",len(class_name1))
print("size of file name",len(file_name))
print("forming matrix -8")

#data,class_name1,first_time,count=featuretoarray1(data,features1, 3, False,count,class_name1)
print(data)
print(len(data))
print("filesnames",fileNames)
#data=[[0.0372626378297882, 0.23106748238629, 0.079174954627949171, 0.102230928556277], [0.0372626378297882, 0.23106748238629, 0.079174954627949171, 0.102230928556277], [0.0372626378297882, 0.23106748238629, 0.079174954627949171, 0.102230928556277], [0.0372626378297882, 0.23106748238629, 0.079174954627949171, 0.102230928556277]]
#print(class_name1)
#class_name1=[1,1,2,2]
#ll=np.array(data)
#ll.astype(np.float32)
#model=KNeighborsRegressor(n_neighbors=2)
#model=KMeans(n_clusters=2)
#model=SVC()
#model=LogisticRegression()
#model=GaussianNB()
#model=AdaBoostClassifier()

#model.fit(data, class_name1)
#print(model.predict(data[0]))
#print(model.score(data,class_name1))
#class_name1=[1,1,2,2]

#min_max_scaler=preprocessing.MinMaxScaler(feature_range=(-1000, 10))
features_scaled=data
'''
print(features_scaled.shape)
print(features_scaled.min(axis=0))
print(features_scaled.max(axis=0))
scatter(features_scaled[:,0],features_scaled[:,1])
'''
#labels=model.fit_predict(features_scaled)
#print(labels)




print("checking for the song",file_name[2])
print("original class",class_name1[2])

model=GMM(n_components=7)
model.fit(data)
print("EM predict class",model.predict(data[2]))
print("EM predict class",model.predict_proba(data[2]))
#print("EM score",model.score(data, class_name1))

model=SVC(probability=True)
#model.fit(data,class_name1)
#print("SVM predict class",model.predict(data[2]))
#print("SVM predict class",model.predict_proba(data[2]))
#print("SVM score",model.score(data, class_name1))

model=GaussianNB()
#model.fit(data,class_name1)
#print("Gausian Naive predict class",model.predict(data[2]))
#print("Gausian Naive predict class",model.predict_proba(data[2]))
#print("GNB score",model.score(data, class_name1))

model=KMeans(n_clusters=7)
model.fit(data)
#print("Kmeans predict class",model.predict(data[2]))
#print("Kmeans score",model.score(data, class_name1))
#print("current model fit",model.aic(data))
#scatter(data[:2,1],data[2:,2],c='b')
#model=affinity_propagation(damping=0.5)
#lables=model.fit_predict(features_scaled)
#print(lables)
b = open('sample.csv', 'w', newline='')
a = csv.writer(b)
data.append(class_name1)
data.append(file_name)
a.writerows(data)
b.close()