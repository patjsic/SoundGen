#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import random
import scipy.io.wavfile
import IPython as ip
import matplotlib.pyplot as plt
import wave
from sympy.utilities.iterables import multiset_permutations
import struct


# In[9]:


#Sample rate and duration of the sound
rate = 96000
duration = 0.5
t = np.linspace(0, duration, int(rate * duration))
np.set_printoptions(threshold=np.inf)

#Random Seed
seed = 10

def norm(v):
    return 2 * (v - np.mean(v)) / np.ptp(v)


class ToneBlock:
    
    #Takes in silence duration, tone duration, the fundamental freq, and number of overtones
    def __init__(self, freq, n_freq, sil=0.4, dur=0.1,):
        self.sil = sil
        self.dur = dur
        self.freq = freq
        self.nfreq = n_freq
    
    #Generates a single tone with a silence after
    def generate_tone(self, freq):
        tone = np.sin(freq * 2.0 * np.pi * np.linspace(0,self.dur,int(rate * self.dur)),dtype=np.float32)
        ramplen = int(0.01 * rate)
        window = np.ones(int(self.dur * rate), dtype=np.float32)
        hann = np.hanning(2 * ramplen).astype(np.float32)
        window[:ramplen] = hann[:ramplen]
        window[-ramplen:] = hann[-ramplen:]
        return np.concatenate([window * tone, np.zeros(int(rate*self.sil), dtype=np.float32)])
    
    #Takes in boolean array of order of frequencies and returns a harmonic stack, 
    #e.g. [0, 1, 1] returns stack of 3 tones with missing fundamental frequency.
    def generate_block(self, farray):
        block = np.zeros(len(farray))
        count = 0
        for i in farray:
            if i:
                block[count] = self.freq * (count + 1)
            count += 1
        return block
    

    
    #Takes in boolean array and generates the tone using generate_block
    def generate_toneblock(self, boolarr):
        sum_series = np.zeros(len(boolarr), dtype = object)
        blok = self.generate_block(boolarr)
        freqs = [self.freq * (i+1) for i, present in enumerate(boolarr) if present]
        return sum(self.generate_tone(freq) for freq in freqs) / len(freqs)

    #Creating random array of boolean arrays of desired length (currently configured to ignore missing midtones)       
    def bool_gen(self):
        big_bool = []
        dim = self.nfreq
        array = np.zeros(dim)
        for i in range(dim):
            array[:i] = np.ones(i)
            for p in multiset_permutations(array):
                big_bool += [p]
        return big_bool

def save_wav(filename_mark,filename_audio, audio):
    mark = np.zeros(len(audio))
    mark[np.nonzero(audio)] = 1
    plt.figure()
    plt.plot(mark)
    scipy.io.wavfile.write(filename_audio, rate, audio)
    scipy.io.wavfile.write(filename_mark,rate,mark)
    
def save_wavstereo(filename, audio):
    mark = np.zeros(len(audio))
    mark[np.nonzero(audio)] = 1
    x = np.array([mark,audio])
    plt.figure()
    plt.plot(mark)
    scipy.io.wavfile.write(filename, rate, x.T)

#Splits array into blocks of 0.5s and shuffles them
def randomizer(tonearray):
    split = np.array(np.array_split(tonearray, 56))
    np.random.shuffle(split)
    x = np.concatenate(split)
    return x
    
    
#Call bool_gen to create all possible stacks (without missing midtones).
#Then create a harmonic stack object for each fundamental frequency. 
def main():
    j = 0
    tone0 = ToneBlock(500, 5)
    tone1 = ToneBlock(1100, 5)
    tone2 = ToneBlock(2500, 5)
    tone3 = ToneBlock(7000, 5)
    tone = [tone0, tone1, tone2, tone3]
    
    pure_tone = []
    boole = [
        [1,1,1,1,1],
        [1,1,1,1,0],
        [1,1,1,0,0],
        [1,1,0,0,0],
        [0,0,0,1,1],
        [0,0,1,1,1],
        [0,1,1,1,1],
        [0,0,1,1,0],
        [0,1,1,0,0],
    
        #Pure Tones
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,0,1],
    ]
    
    #Generate harmonic stacks using boole
    toneblock0 = np.array([[x,  tone0.freq, tone0.generate_toneblock(i)] for x,i in enumerate(boole)])
    toneblock1 = np.array([[x,  tone1.freq, tone1.generate_toneblock(i)] for x,i in enumerate(boole)])
    toneblock2 = np.array([[x,  tone2.freq, tone2.generate_toneblock(i)] for x,i in enumerate(boole)])
    toneblock3 = np.array([[x, tone3.freq, tone3.generate_toneblock(i)]for x,i in enumerate(boole)])
    
    #Append toneblock values
    tonecat = np.concatenate((toneblock0, toneblock1,toneblock2, toneblock3))
    
    new_t = np.linspace(0, duration * 14, int(rate * duration * 14))
    full_tonecat = np.repeat(tonecat, 8, 0)

    #plotcat = np.concatenate(tonecat)
    np.random.shuffle(full_tonecat)
    toneplot = [full_tonecat[i][2] for i in range(len(full_tonecat))]
    
#     for cat in tonecat:
        
#         plt.title('$f_{o}$ =' + str(tone[j].freq))
#         plt.plot(new_t, cat)
#         plt.axis([0.505, 0.51, -1, 1])
#         plt.figure()
#         j += 1
    
    #Insert a second of silence at the start
    toneplot = np.insert(toneplot, 0, np.zeros(rate))

    #plt.plot(new_t, tonecat[0])
    #plt.plot(new_t, tonecat[1]) #To show waves aren't out of phase
    #plt.title('$f_{o}$ = 550 Hz and $f_{o} = 1000 Hz$')
    #plt.axis([0.0, 0.6, -1, 1])
    plt.figure()
    plt.plot(toneplot)
    toneparam = [[full_tonecat[i][0],full_tonecat[i][1]] for i in range(len(full_tonecat))]
    #save_wav('mark96k.wav','test96k.wav',toneplot)
    save_wavstereo('test96k2', toneplot)
    f = open("params96k2.txt","w+")
    f.write("seed : %s" %seed  + "\n tone array with corresponding frequency: %s" %toneparam)
    f.close
    
main()



# In[ ]:




