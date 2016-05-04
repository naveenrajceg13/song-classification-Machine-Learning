import os
import csv
import rpy2.robjects as ro
import numpy as np
#from rpy2.robjects.packages import importr

b = open('C:\project\data_change_frequency.csv', 'w', newline='')
a = csv.writer(b)
data=[[]]
count=0
def Getfilesinfolder(path):
    list=[]
    for dir_entry in os.listdir(path):
        dir_entry_path = os.path.join(path, dir_entry)
        print(dir_entry_path)
        if os.path.isfile(dir_entry_path):
            list.append(dir_entry_path)
    return list;
def getcentriodandspread(samples):
    ind = (np.arange(1, len(samples) + 1)) * (400/(2.0 * len(samples)))
    Xt = samples.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + 13
    
    # Centroid:
    C = (NUM / DEN)
    
    # Spread:
    S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)
    
    # Normalize:
    print(C)
    print(S)
    C = C / (440 / 2.0)
    S = S / (440 / 2.0)
    print(C)
    print(S)

def getfilesintofile(list,count,class_val):  
    for files in list:
        sub_count=0
        bool=True
        with open(files,'r') as my_file:
            try:
                s=my_file.read()
            except:
                bool=False
                pass
            if sub_count==0:
                 data[count+1].append(class_val)
            value=s.split("\t")
            sample=np.array(value)
            getcentriodandspread(sample)
            break;
            for each in value:
                if(sub_count==0):
                    sub_count=sub_count+1
                    continue
                if(sub_count>1500 and sub_count<=2500):
                    value1=each.split("\n")
                    for ea in value1:
                        value1=ea
                        break
                    value1=value1.split("A=")
                    for ea in value1:
                        value1=ea
                    data[count+1].append(value1)
                    value1=each.split("d=")
                    for ea in value1:
                        value1=ea
                    data[count+1].append(value1)
                if bool==True:
                    if(sub_count>0 and sub_count<=1000):
                        if count==0:
                            pass
                            if sub_count==1:
                                data[0].append("Class")
                            data[0].append("A "+str(sub_count))
                            data[0].append("D "+str(sub_count))
                    sub_count=sub_count+1
                    bool=True
                if(sub_count>3500):
                    break
        temp=[]
        data.append(temp)
        count=count+1
    return count   
        
list=Getfilesinfolder("C:\project\music_feature\country")

temp=[]
data.append(temp)
count=getfilesintofile(list,0,"COUNTRY")
'''
list=Getfilesinfolder("C:\project\music_feature\EDM")
count=getfilesintofile(list,count,"EDM")
list=Getfilesinfolder("C:\project\music_feature\Hip-Hop")
count=getfilesintofile(list,count,"Hip-Hop")
list=Getfilesinfolder("C:\project\music_feature\jazz")
count=getfilesintofile(list,count,"jazz")
list=Getfilesinfolder("C:\project\music_feature\Rock")
count=getfilesintofile(list,count,"Rock")
a.writerows(data)
'''