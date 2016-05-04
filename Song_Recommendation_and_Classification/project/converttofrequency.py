'''
Created on Apr 14, 2016

@author: NAVE
'''
import os
import csv
import rpy2.robjects as ro
#from rpy2.robjects.packages import importr

b = open('C:\project\data_frequency.csv', 'w', newline='')
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
            value=s.split("\n")
            
            for each in value:
                if(sub_count==0):
                    sub_count=sub_count+1
                    continue
                if(sub_count>3500 and sub_count<=5500):
                    data[count+1].append(each)
                if bool==True:
                    if(sub_count>0 and sub_count<=2000):
                        if count==0:
                            pass
                            if sub_count==1:
                                data[0].append("Class")
                            data[0].append("F "+str(sub_count))
                    sub_count=sub_count+1
                    bool=True
                if(sub_count>5500):
                    break
        temp=[]
        data.append(temp)
        count=count+1
    return count   
        
list=Getfilesinfolder("C:\project\music_fre\country")
temp=[]
data.append(temp)
count=getfilesintofile(list,0,"COUNTRY")
list=Getfilesinfolder("C:\project\music_fre\EDM")
count=getfilesintofile(list,count,"EDM")
list=Getfilesinfolder("C:\project\music_fre\Hip-Hop")
count=getfilesintofile(list,count,"Hip-Hop")
list=Getfilesinfolder("C:\project\music_fre\jazz")
count=getfilesintofile(list,count,"jazz")
list=Getfilesinfolder("C:\project\music_fre\Rock")
count=getfilesintofile(list,count,"Rock")
a.writerows(data)