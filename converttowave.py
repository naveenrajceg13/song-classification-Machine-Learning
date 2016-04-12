import pydub

#sound = pydub.AudioSegment.from_mp3("test.mp3")
#sound = sound.set_channels(1)
#sound.export("test1.wav", format="wav")


import os
from scipy.io.wavfile import read, write
from scipy.fftpack import rfft, irfft
from scipy.fftpack import fft
import string
import numpy
import numpy as np
import matplotlib.pyplot as plt

def Getfilesinfolder(path):
    list=[]
    for dir_entry in os.listdir(path):
        dir_entry_path = os.path.join(path, dir_entry)
        print(dir_entry_path)
        if os.path.isfile(dir_entry_path):
            list.append(dir_entry_path)
    return list;
def getfilesintofile(list):  
    for files in list:
        sound = pydub.AudioSegment.from_mp3(files)
        sound = sound.set_channels(1)
        value=str.replace(files, 'music', 'music_wav')
        file_nmae=value[:-4]+".wav"
        sound.export(file_nmae, format="wav") 
        
def getfilesintofrequency(list):  
    for files in list:
        sum=0
        max=0
        flag=True
        rate, input = read(files)
        transformed = rfft(input,50000)
       # value=str.replace(files, 'music_wav', 'music_frequency')
        #file_nmae=value[:-4]+".dat"
        #file4 = open(file_nmae, 'w+')
        for each in transformed:
            #file4.write(str(each))
            sum=sum+int(each)
            print(sum)
            if(flag):
                max=int(each)
                flag=False
            if(max<int(each)):
                max=int(each)
            if(sum>=0 and sum<=1):
                print("-----------------")
                break
            #file4.write("\n")
        print("max is",max)
        plt.plot(transformed)
        plt.show()
        break            
         
        
'''
list=Getfilesinfolder("C:\project\music\country")
getfilesintofile(list)
list=Getfilesinfolder("C:\project\music\EDM")
getfilesintofile(list)
list=Getfilesinfolder("C:\project\music\Hip-Hop")
getfilesintofile(list)
list=Getfilesinfolder("C:\project\music\jazz")
getfilesintofile(list)
list=Getfilesinfolder("C:\project\music\Rock")
getfilesintofile(list)
'''
list=Getfilesinfolder("C:\project\music_wav\country")
getfilesintofrequency(list)
'''
list=Getfilesinfolder("C:\project\music_wav\EDM")
getfilesintofrequency(list)
list=Getfilesinfolder("C:\project\music_wav\Hip-Hop")
getfilesintofrequency(list)
list=Getfilesinfolder("C:\project\music_wav\jazz")
getfilesintofrequency(list)

list=Getfilesinfolder("C:\project\music_wav\Rock")
getfilesintofrequency(list)
'''