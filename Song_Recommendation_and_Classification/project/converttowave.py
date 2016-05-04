import pydub
import gc
#sound = pydub.AudioSegment.from_mp3("test.mp3")
#sound = sound.set_channels(1)
#sound.export("test1.wav", format="wav")

import xlrd
#book=xlrd.open_wo6.,.....,........rkbook("sample.xls","rw")
import os
from scipy.io.wavfile import read, write
from scipy.fftpack import rfft, irfft
from scipy.fftpack import fft
import string
import numpy
import numpy as np
import matplotlib.pyplot as plt
from tkinter.tix import INCREASING
import csv
import struct
from wave_sample import opens
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from scipy.fftpack.realtransforms import dct
from pyAudioAnalysis import utilities
from getfeatures import getcentriodandspread,getfileintoframe

#b = open('C:\project\data_change_frequency.csv', 'w', newline='')
#a = csv.writer(b)
data=[[]]
count=0
def Getfilesinfolder(path):
    list=[]
    for dir_entry in os.listdir(path):
        dir_entry_path = os.path.join(path, dir_entry)
        try:
            #print(dir_entry_path)
            if os.path.isfile(dir_entry_path):
                list.append(dir_entry_path)
        except:
            pass
    return list;

def getfilesintofile(list):  
    for files in list:
        try:
            sound = pydub.AudioSegment.from_mp3(files)
            sound = sound.set_channels(1)
            value=str.replace(files, 'music', 'music_wave')
            file_nmae=value[:-4]+".wav"
            file_nmae1=file_nmae.replace(',','123')
            try:
                 index=file_nmae.index(',')
                 print(index)
                 print(file_nmae1)
            except:
                pass
            sound.export(file_nmae1, format="wav") 
        except:
            pass
        
def getfilesintofrequency(list,count,class_value,first_sheet): 
    
    frequency_data=[]
    for files in list:
        distance=0
        end=0
        cur_value=0
        start=0
        intial=True
        next_intial=False
        decreasing=False
        INCREASING=False
        not_visited=True
        transformed=""
        not_done=True
        #try:
        rate, input = read(files)
        
        transformed = rfft(input,rate)
        not_done=False
        sample=np.array(transformed)
        value=str.replace(files, 'music_wav', 'music_ext_1')
        file_nmae_feature1=value[:-4]+"_extracted_main.dat"
        value=str.replace(files, 'music_wav', 'music_ext_2')
        file_nmae_feature2=value[:-4]+"_extracted_chroma.dat"
        value=str.replace(files, 'music_wav', 'music_ext_3')
        file_nmae_feature3=value[:-4]+"_extracted_mcff.dat"
        file6= open(file_nmae_feature1, 'w+')
        file7= open(file_nmae_feature2, 'w+')
        file8= open(file_nmae_feature3, 'w+')
        getfileintoframe(files, 1500,file6,file7,file8)
        getcentriodandspread(sample,file6)
        #except:
            #pass
            #print('error')
        if(not_done==False):
            value=str.replace(files, 'music_wav', 'music_frequency')
            file_nmae=value[:-4]+".dat"
            value=str.replace(files, 'music_wav', 'music_feature')
            file_nmae_feature=value[:-4]+"_feature.dat"
            file_nmae_feature=value[:-4]+"_feature_ch.dat"
            file4 = open(file_nmae, 'w+')
            file5= open(file_nmae_feature, 'w+')
            
            if(count==1):
                data[0].append("class")
            data[count].append(class_value)
            sub_count=1
            col_count=1
        for each in transformed:
            not_visited=True
            file4.write(str(each))
            frequency_data.append(str(each))
            file4.write("\n")
            
            if(INCREASING and not_visited):
                not_visited=False
                if(cur_value<int(each)):
                    cur_value=int(each)
                    INCREASING=True
                    decreasing=False
                    distance=distance+1
                    
                else:
                    ampti=cur_value-start
                    file5.write("d="+str(distance)+"\t"+"A="+str(ampti))
                    if(sub_count>=1 and sub_count<=850):
                        #data[count].append(str(distance))
                        #data[count].append(str(ampti/distance))
                        if(count==1):
                            pass
                            #data[0].append("D"+str(col_count))
                            #data[0].append("A"+str(col_count))
                        #col_count=col_count+1
                    file5.write("\n")
                    distance=1
                    start=int(each)
                    INCREASING=False
                    decreasing=False
                    next_intial=True
                    cur_value=start
                    #sub_count=sub_count+1
            if(decreasing and not_visited):
                not_visited=False
                if(cur_value>int(each)):
                    cur_value=int(each)
                    INCREASING=False
                    decreasing=True
                    distance=distance+1
                    
                else:
                    ampti=start-cur_value
                    file5.write("d="+str(distance)+"\t"+"A="+str(ampti))
                    if(sub_count>=1 and sub_count<=850):
                        #data[count].append(str(distance))
                        #data[count].append(str(ampti/distance))
                        if(count==1):
                            pass
                            #data[0].append("D"+str(col_count))
                            #data[0].append("A"+str(col_count))
                        #col_count=col_count+1
                    file5.write("\n")
                    distance=1
                    start=int(each)
                    INCREASING=False
                    decreasing=False
                    next_intial=True
                    cur_value=start
                    #sub_count=sub_count+1
            if(next_intial and not_visited):
                not_visited=False
                if(cur_value<int(each)):
                    INCREASING=True
                    decreasing=False
                    distance=distance+1
                    next_intial=False
                    cur_value=int(each)
                    
                if(cur_value>int(each)):
                    INCREASING=False
                    decreasing=True
                    distance=distance+1
                    next_intial=False
                    cur_value=int(each)
                    
            if(intial and not_visited):
                not_visited=False
                start=int(each)
                cur_value=int(each)
                distance=1
                intial=False
                next_intial=True
                
        if(not_done==False):   
            array=plt.plot(transformed)
            #plt.show()
            #print(array)
            file_nmae=value[:-4]+".png"
            #plt.savefig(file_nmae)
            count=count+1
            temp=[]
            data.append(temp)
            gc.collect()
            xvalues = array[0].get_xdata()
            yvalues = array[0].get_ydata()
             
    #first_sheet.write(frequency_data)  
    return count
      
list=Getfilesinfolder("C:\project\songs\music\A")
getfilesintofile(list)
list=Getfilesinfolder("C:\project\songs\music\B")
getfilesintofile(list)
list=Getfilesinfolder("C:\project\songs\music\C")
getfilesintofile(list)
list=Getfilesinfolder("C:\project\songs\music\D")
getfilesintofile(list)
list=Getfilesinfolder("C:\project\songs\music\E")
getfilesintofile(list)
list=Getfilesinfolder("C:\project\songs\music\F")
getfilesintofile(list)
list=Getfilesinfolder("C:\project\songs\music\G")
getfilesintofile(list)
'''
first_sheet=book.sheet_by_index(0)
list=Getfilesinfolder("C:\project\music_wav\country\E\A")
temp=[]
data.append(temp)
count=getfilesintofrequency(list,1,"",first_sheet)
plt.clf()
list=Getfilesinfolder("C:\project\music_wav\country\F\B")
count=getfilesintofrequency(list,count,"EDM",first_sheet)
plt.clf()
list=Getfilesinfolder("C:\project\music_wav\country\F\C")
count=getfilesintofrequency(list,count,"country",first_sheet)
plt.clf()
list=Getfilesinfolder("C:\project\music_wav\country\F\A")
count=getfilesintofrequency(list,count,"country",first_sheet)
plt.clf()
list=Getfilesinfolder("C:\project\music_wav\country\F\E")
count=getfilesintofrequency(list,count,"EDM",first_sheet)
plt.clf()
list=Getfilesinfolder("C:\project\music_wav\country\F\F")
count=getfilesintofrequency(list,count,"EDM",first_sheet)
plt.clf()
a.writerows(data)
b.close()
print(data[0])
print(data[1])
print(data[2])
print(data[3])
print(data[4])
print(data[5])
'''