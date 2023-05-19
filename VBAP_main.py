# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:19:58 2023

@author: Téo Di Bisceglie, Matthias Blondet
"""

import numpy as np
import matplotlib.pyplot as plt
import ClassVBAP as VBAP

from scipy import signal
from scipy.io import wavfile

plt.rc('lines', linewidth=3)
plt.rc('font', size=20)
plt.rc('axes', linewidth=1.5, labelsize=20)
plt.rc('legend', fontsize=20)
plt.rcdefaults()

scaling_2D = 1/np.loadtxt("Correction 2D.txt")
scaling_3D = 1/np.loadtxt("Correction 3D.txt")

Fs = 48000

#%% 2D SETUP

n = 8
phi_s = np.arange(0,360,360/n)

VB2D = VBAP.VBAP(phi_s,Fs,scaling=scaling_2D)

#%% 3D SETUP

space = space = np.zeros((17,2))
elevation = 41.8

for i in range(16): # Configuration of the experimental set
    
    if i <8:
        
        space[i] = np.concatenate(([0+i*45],[0]))
    
    elif i>=8 and i < 12:
        
        space[i] = [315+(+i-8)*90,elevation]
     
    else:
        
        space[i] = [225+(i-12)*90,-elevation]

space[-1] = [0,90]

VB3D = VBAP.VBAP(space,Fs=48000,scaling=scaling_3D)

#%% Signals functions

def sine(f,duration,Fs,g=1):
    
    t = np.arange(0,int(duration*Fs))/Fs
    
    sine = g*np.sin(2*np.pi*f*t)
    
    return(Fs, sine)

def readwave(name,tstart=0,tend=3):
    
    Fs, data = wavfile.read(name+".wav")

    data = data[int(tstart*Fs):int(tend*Fs)]
    
    datar = None

    if data.dtype=='int16':
        
        norm = 2**15
        
    if data.dtype=='int32':
        
        norm = 2**31
        
    if data.dtype=='int64':
        
        norm = 2**63
        
    datal = data[:,0]/norm
        
    if data.ndim==2:
        
        datar = data[:,1]/norm
            
    return(Fs, datal,datar)

#%% Wav file

filename = "Karnivool Deadman"
tstart = 0
tstop = 20

Fsr,left,right = readwave(filename,tstart,tstop) # If mono only left

num = round(len(left)*float(48000)/Fsr)
left = signal.resample(left, num)
right = signal.resample(right, num)

#%% 2D Sine palying

Fs, sinus = sine(f=600,duration=3,Fs=Fs)

''' 1 turn in each direction
moving_angles = np.arange(0,361,5) # virtual sources with moving angles
moving_angles = np.concatenate((moving_angles,np.arange(0,-361,-5)))
'''

moving_angles = np.arange(0,361,10) # 1 trigonometric turn
moving_angles = [20] # Fixed angle

gains,indices = VB2D.VBAP_2D(moving_angles,0.7)

out = VB2D.NS_2d_mapping(gains, indices, data=sinus,Fs=Fs,Wsignal = 'Rectangle',linear=True)

# VB2D.play(out)

#%% 2D WAV playing

moving_angles = np.arange(0,361,10) # Moving angles
# moving_angles = [-18] # Fixed angle

gains,indices = VB2D.VBAP_2D(moving_angles,0.7)
out = VB2D.NS_2d_mapping(gains, indices, data=left,Fs=Fs,Wsignal = 'Rectangle',linear=True)

# VB2D.play(out)

#%% 3D Sine playing

Fs, sinus = sine(f=600,duration=3,Fs=Fs)

moving_angles = [[20,20]] # Fixed angle

az = np.arange(0,361,5)
el = np.full_like(az, 31)
moving_angles = np.array([az,el]).T # Moving

gains,indices = VB3D.VBAP_3D(moving_angles,0.7)
out = VB3D.NS_3d_mapping(gains, indices, data=sinus,Fs=Fs,Wsignal = 'Tukey',linear=True)

# VB3D.play(out)

#%% 3D WAV playing

moving_angles = [[20,20]] # Fixed angle

az = np.arange(0,361,5)
el = np.full_like(az, 31)
moving_angles = np.array([az,el]).T # Moving

gains,indices = VB3D.VBAP_3D(moving_angles,0.7)
out = VB3D.NS_3d_mapping(gains, indices, data=left,Fs=Fs,Wsignal = 'Rectangle',linear=True)

# VB3D.play(out)

#%% Plot of output signal in 2D

plt.close('all')

fig, ax = plt.subplots(figsize=(9,5),dpi=180,tight_layout=True)

for i in range(VB2D.n):
    
    ax.plot(np.arange(0,len(out))/Fs, out[:,i], label = f"voie {i}",alpha=0.8)

ax.set_xlabel("Temps (s)")
ax.set_ylabel("Signal de sortie")

ax.grid()
ax.legend()

plt.show()

#%% Plot of output signal in 3D

plt.close('all')

fig, ax = plt.subplots(figsize=(9,5),dpi=180,tight_layout=True)

for i in range(VB3D.n): 
    
    ax.plot(np.arange(0,len(out))/Fs, out[:,i], label = f"voie {i}",alpha=0.8)

ax.set_xlabel("Temps (s)")
ax.set_ylabel("Signal de sortie")

ax.grid()
ax.legend()

plt.show()