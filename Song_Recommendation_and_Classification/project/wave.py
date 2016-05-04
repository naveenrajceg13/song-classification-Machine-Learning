from numpy.fft import *
from numpy.fft import fftpack
from numpy import *
import math
import struct
import numpy as np
import matplotlib.pyplot as plt
from test.test_buffer import numpy_array
from scipy.signal import waveforms
from wave_sample import opens
from scipy.fftpack import fft
import subprocess
import pydub
"""
Turn a wave file into an array of ints
Wav file should not contain Metadata
"""
def get_samples(file):
    file1 = open('audio_file1'+'.dat', 'w+')
    waveFile = opens(file, 'rb')
    samples = []
    print(waveFile)
    print("Gets total number of frames");
    length = waveFile.getnframes()
    print("length is",length)
    # Read them into the frames array
    for i in range(0,length):
        waveData = waveFile.readframes(1)
       # data = struct.unpack("%ih"%2, waveData)
        filename = "./outfile.dat.npy"
        np.save(filename, np.array(waveData))
        z = np.load(filename).tolist()
        #print("z is: " + str(z))
        data=struct.unpack("%ih"%1,waveData)
        # After unpacking, each data array here is actually an array of ints
        # The length of the array depends on the number of channels you have
        # Drop to mono channel
        #file1.write(str(data[0]))
        #file1.write(" ")
        samples.append(int(data[0]))
        
    samples = np.array(samples)
   # file.write(samples)
    return samples

def energy(samples):
    return sum([x**2 for x in samples])

'''
#file="beep.wav"
file="beep.wav"
print("sending data to get sample")
samples = get_samples(file)
print(type(samples))
file4 = open('fft_values'+'.dat', 'w+')
print("samples",samples)
freq_domain = fft(samples)
print("fft",freq_domain)
for each_val in freq_domain:
    print("hi",each_val)
    file4.write(str(each_val))
    file4.write("\n")
print("energy",energy(samples))
#plt.plot(samples)    # your return from get_samples()
#plt.show()
# Compute the FFT of the samples (gets the frequencies used)
#plt.plot(freq_domain)
#plt.show()
 
# Compute energy of samples by squaring and summing
'''