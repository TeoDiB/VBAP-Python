# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:44:12 2023

@author: teodi
"""

import numpy as np
import matplotlib.pyplot as plt

import ClassVBAP as VBAP
from matplotlib.patches import Patch
plt.rcdefaults()

Fs = 48000
plt.close('all')

space = space = np.zeros((17,2))
elevation = 41.8

for i in range(16):
    
    if i <8:
        
        space[i] = np.concatenate(([0+i*45],[0]))
    
    elif i>=8 and i < 12:
        
        space[i] = [315+(+i-8)*90,elevation]
     
    else:
        
        space[i] = [225+(i-12)*90,-elevation]

space[-1] = [0,90]

VB3D = VBAP.VBAP(space,Fs=48000)

aziRes = 2
elevRes = 2

Elev, Azi = np.meshgrid(np.arange(-90, 91, elevRes), np.arange(0, 361, aziRes))

src_dirs3D = np.column_stack((Azi.ravel(), Elev.ravel()))

gains3D, ind3D  = VB3D.VBAP_3D(src_dirs3D,C=0.95)

matrix = np.zeros((len(gains3D),17))

for i in range(len(gains3D)):
    
    i1 = int(ind3D[i,0])
    i2 = int(ind3D[i,1])
    i3 = int(ind3D[i,2])
    
    g1 = gains3D[i,0]
    g2 = gains3D[i,1]
    g3 = gains3D[i,2]
    
    matrix[i,i1] = g1
    matrix[i,i2] = g2
    matrix[i,i3] = g3

nAz, nElev = Azi.shape

X = np.cos(Elev*np.pi/180)*np.cos(Azi*np.pi/180)
Y = np.cos(Elev*np.pi/180)*np.sin(Azi*np.pi/180)
Z = np.sin(Elev*np.pi/180)

fig = plt.figure(dpi=180)

ax = fig.add_subplot( projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.grid()

fcolors = ['#000080', '#0000FF', '#0080FF', '#00FFFF', '#00FF80',
                 '#80FF00', '#FFFF00', '#FF8000', '#FF0000', '#800000',
                 '#800080', '#FF00FF', '#808080', '#C0C0C0','#E30B5D',
                 '#008000', '#4a4a4a']

legend_patches = []

for nl in range(VB3D.n):
    
    gains_grid_nl = matrix[:,nl].reshape(nAz, nElev)

    legend_patches.append(Patch(color=fcolors[nl], label=f'Voie {nl}'))
    
    ax.plot_surface(gains_grid_nl*X, gains_grid_nl*Y, gains_grid_nl*Z,label=f"Voie {nl}",color=fcolors[nl])

ax.legend(handles=legend_patches,loc="upper left")

ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_zticks([-1, -0.5, 0, 0.5, 1])

plt.tight_layout()
plt.show()