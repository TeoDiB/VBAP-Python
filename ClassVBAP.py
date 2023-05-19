# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:21:01 2023

@author: Téo Di Bisceglie, Matthias Blondet
"""

import numpy as np

import SoundCard as SCmod

from scipy import signal
from scipy.spatial import ConvexHull

class VBAP:
    
    def __init__(self,s_angles,Fs=48000,scaling=None):
        """

        Parameters
        ----------
        s_angles : float
            Angular positions of the sources in degree. 1 Angle means 2D setup, 
            
            In 2D angles have to be entered in the right order from source 0 to n
            in trigonomertic direction. 
            
            2 angles means 3D setup (angles are in spherical convention
                                     rayon/longitude/latitude in degrees))
            In 3D triangularisation is done so not important.
        
        Fs : int, optional
            Sampling frequency of the system. The default is 48000.
            
        scaling : array, optional
            Linear gains to apply to each source to normalize each distant.
            Warning : can cause cliping so scale relative to the furthest source.
        """

        if len(np.shape(s_angles)) ==2: # 3D setup == triangulation
                                
            azim = np.deg2rad(s_angles[:,0])
            elev = np.deg2rad(s_angles[:,1])
            
            self.x = np.cos(elev)*np.cos(azim)
            self.y = np.cos(elev)*np.sin(azim)
            self.z = np.sin(elev)
            
            self.v = np.transpose(np.array([self.x,self.y,self.z]))

            self.hull = ConvexHull(np.transpose(np.array([self.x,self.y,self.z])), incremental=True, qhull_options=None)
            
            self.triangles = self.hull.simplices        
            self.s_angles = s_angles
            
            self.Lmat = [np.zeros((3,3)) for i in range(len(self.triangles))]
            
            for count, tri in enumerate(self.triangles):
                
                v1 = self.v[tri[0]]
                v2 = self.v[tri[1]]
                v3 = self.v[tri[2]]
                
                self.Lmat[count] = np.array([v1,v2,v3])
                
        else: # 2D setup
        
            self.s_angles = np.deg2rad(s_angles)
        
        self.n = len(s_angles) # Number of sources       
        
        self.SC = SCmod.SoundCard()
        self.SC.Fs = Fs
        self.SC.SCnochoice()
        self.SC.mapping([x for x in range(18)], [x for x in range(self.n)])
        
        if scaling is None: # No Sources scaling
            
            self.scaling = np.ones(self.n)
            
        else: # not uniform distances = sources scaling
            
            self.scaling = scaling
        
    def VBAP_2D(self,VS_angles,C=0.5):
        """ 2D VBAP
        Calculation of gain factors for virtual source(s) at angle(s) VS_angles

        Parameters
        ----------
        VS_angle : Array
            Angular positions of the virtual source in degrees.
        C : float
            normalisation constant (recomended in [0,1])

        Returns
        -------
        None.
        """
        
        L = np.zeros((self.n+1,2))

        Lmat = [np.zeros((2,2)) for i in range(self.n)]
        
        for count, phi in enumerate(self.s_angles):
            
            l = np.array([np.cos(phi),np.sin(phi)])
            
            L[count,] = l
            
            if count == 0:
                
                L[-1] = l
                
        Lmat = [L[i:i+2] for i in range(self.n)]

        L_matinv = [np.linalg.inv(i) for i in Lmat]
        
        gains = np.zeros((len(VS_angles),2))
        indices = np.zeros((len(VS_angles),2))
        
        for count, VS_angle in enumerate(VS_angles):
        
            phi = np.deg2rad(VS_angle) # Virtual angle
            p = np.array([np.cos(phi),np.sin(phi)])
    
            g = [np.transpose(p)@L_matinv[i] for i in range(self.n)]
               
            for i in range(len(g)):
                
                if all(val >= 0 for val in g[i]):
                    
                    g = g[i]
                    
                    if i==(self.n-1):
                        right_index = self.n-1
                        left_index = 0
                    else:    
                        right_index = i
                        left_index = i+1
                    
                    g_scaled = np.sqrt(C)/np.sqrt((g[0]**2+g[1]**2))*g
                    
                    gains[count] = g_scaled # [right, left]
                    indices[count] = [right_index,left_index]
                    
                    break
                
                elif i == len(g)-1:
                    
                    print("Problème d'inversion")
                    
                    #print(np.linalg.cond(A),np.inf)) # Conditionnement of A
                
        return gains, indices
    
    def VBAP_3D(self,VS_angles,C=0.5):
        """ 3D VBAP
        Calculation of gain factors for virtual source(s) at angle(s) VS_angles

        Parameters
        ----------
        VS_angle : Array
            Angular positions of the virtual source in degrees.
            (spherical convention rayon/longitude/latitude in degrees)
        C : float
            normalisation constant (recomended in [0,1])

        Returns
        -------
        None.
        """
        
        L_matinv = [np.linalg.inv(i) for i in self.Lmat]
        
        gains = np.zeros((len(VS_angles),3))
        indices = np.zeros((len(VS_angles),3))
        
        for count, VS_angle in enumerate(VS_angles):
        
            phi = np.deg2rad(VS_angle[1]) # Virtual angle
            theta = np.deg2rad(VS_angle[0])
            
            p = np.array([np.cos(phi)*np.cos(theta),np.cos(phi)*np.sin(theta),np.sin(phi)])
            
            g = [np.transpose(p)@L_matinv[i] for i in range(len(self.triangles))]
                
            for i in range(len(g)):
                
                if all(val >= 0 for val in g[i]):
                    
                    g = g[i]
                    
                    g_scaled = np.sqrt(C)/np.sqrt((g[0]**2+g[1]**2+g[2]**2))*g
                    
                    indexes = self.triangles[i]
                    
                    gains[count] = g_scaled
                    
                    indices[count] = indexes
                    
                    break
                
                elif i == len(g)-1:
                    
                    print("Problème d'inversion")
                    
                    #print(np.linalg.cond(A,np.inf)) # Conditionnement of A
        
        return gains, indices
        
    
    def NS_2d_mapping(self,gains,indices,data,Fs, Wsignal = 'Rectangle',linear=False):
        """ 2D mapping for gains and indices calculated with VBAP_2D


        Parameters
        ----------
        gains : array of shape (n_angles,2)
            Gain factors calculated with VBAP_2D. n_angles the number of angles to compute.
        indices : array of shape (n_angles,2)
            Indices of the source used. Comes from VBAP_2D
        data : 1D array
            The data to map.
        Fs : int
            Sampling fequency of the data.
        Wsignal : str, optional
            Window to apply to the data. The default is 'Rectangle'.
            Can be Tukey, S_Hann or Rectangle.
        linear : Bool, optional
            Linearisation for moving source if True. The default is False.

        Returns
        -------
        out : array
            Mapped data to play with Soundcard.

        """
        
        out = np.zeros((len(data), self.n))
        
        if Wsignal == 'S_Hann':
            Wsignal = signal.windows.hann(len(data))
        
        elif Wsignal.upper() == 'Tukey':
        
            Wsignal = signal.windows.tukey(len(data),0.05)
        else:
            Wsignal = 1
        
        if len(gains)>1:
            
            if linear: # Linéarisation
            
                nmoov = len(data)/(len(gains)-1)
            
                for count, gain in enumerate(gains):
                    
                    right = int(indices[count][0])
                    left = int(indices[count][1])
                    
                    if count ==0:
                        
                        g_old = np.zeros(self.n)
                        
                        g_old[right] = gain[0]
                        g_old[left] = gain[1]
                        
                    else:
                    
                        gtemp = np.zeros(self.n)
                        
                        gtemp[right] = gain[0]
                        gtemp[left] = gain[1]
                        
                        a = (gtemp-g_old)/(nmoov) # coefficient directeur
                        b = -a*count*nmoov+ gtemp # ordonnée à l'origine
                        
                        f = np.arange(int((count-1)*nmoov),int((count)*nmoov))
                        f = np.full((self.n,len(f)),f).T
                        f = a*f+np.full(np.shape(f),b)
                        
                        out[int((count-1)*nmoov):int((count)*nmoov),:] = f
                        
                        g_old = gtemp
                        
                        if count == np.shape(gains)[0]-1:
                            
                            out[int((count)*nmoov):int((count+1)*nmoov),right] = gain[0]
                            out[int((count)*nmoov):int((count+1)*nmoov),left] = gain[1]
                        
            else: # Without linearisation
            
                nmoov = len(data)/(len(gains))
            
                for count, gain in enumerate(gains):
            
                    right = int(indices[count][0])
                    left = int(indices[count][1])
                    
                    out[int((count)*nmoov):int((count+1)*nmoov),right] = gain[0]
                    out[int((count)*nmoov):int((count+1)*nmoov),left] = gain[1]
            
            #out = out # enveloppe
            out = out*np.transpose(np.tile(data,(self.n,1))) # enveloppe*signal
              
        else: # one fixed angle 
            
            out[:,int(indices[0][0])] = data*Wsignal*gains[0][0] # RIGHT
            out[:,int(indices[0][1])] = data*Wsignal*gains[0][1] # LEFT
            
        
        out = out*self.scaling
        
        return out
    
    def NS_3d_mapping(self,gains,indices,data,Fs, Wsignal = 'Rectangle',linear=False):
        """ 3D mapping for gains and indices calculated with VBAP_3D

        Parameters
        ----------            
        gains : array of shape (n_angles,3)
            Gain factors calculated with VBAP_3D. n_angles the number of angles to compute.
        indices : array of shape (n_angles,3)
            Indices of the source used. Comes from VBAP_3D
        data : 1D array
            The data to map.
        Fs : int
            Sampling fequency of the data.
        Wsignal : str, optional
            Window to apply to the data. The default is 'Rectangle'.
            Can be Tukey, S_Hann or Rectangle.
        linear : Bool, optional
            Linearisation for moving source if True. The default is False.

        Returns
        -------
        out : array
            Mapped data to play with Soundcard.


        """
        
        out = np.zeros((len(data), self.n))
        
        if Wsignal == 'S_Hann':
            Wsignal = signal.windows.hann(len(data))
        
        elif Wsignal.upper() == 'Tukey':
        
            Wsignal = signal.windows.tukey(len(data),0.05)
        else:
            Wsignal = 1
        
        if len(gains)>1:
            
            if linear: # Linéarisation
            
                nmoov = len(data)/(len(gains)-1)
            
                for count, gain in enumerate(gains):
                    
                    ind1 = int(indices[count][0])
                    ind2 = int(indices[count][1])
                    ind3 = int(indices[count][2])
                    
                    if count ==0:
                        
                        g_old = np.zeros(self.n)
                        
                        g_old[ind1] = gain[0]
                        g_old[ind2] = gain[1]
                        g_old[ind3] = gain[2]
                        
                    else:
                    
                        gtemp = np.zeros(self.n)
                        
                        gtemp[ind1] = gain[0]
                        gtemp[ind2] = gain[1]
                        gtemp[ind3] = gain[2]
                        
                        a = (gtemp-g_old)/(nmoov) # coefficient directeur
                        b = -a*count*nmoov+ gtemp # ordonnée à l'origine
                        
                        f = np.arange(int((count-1)*nmoov),int((count)*nmoov))
                        f = np.full((self.n,len(f)),f).T
                        
                        f = a*f+np.full(np.shape(f),b)
                        
                        out[int((count-1)*nmoov):int((count)*nmoov),:] = f
                        
                        g_old = gtemp
                        
                        if count == np.shape(gains)[0]-1:
                            
                            out[int((count)*nmoov):int((count+1)*nmoov),ind1] = gain[0]
                            out[int((count)*nmoov):int((count+1)*nmoov),ind2] = gain[1]
                            out[int((count)*nmoov):int((count+1)*nmoov),ind3] = gain[2]
                        
            else: # Without linearisation
            
                nmoov = len(data)/(len(gains))
            
                for count, gain in enumerate(gains):
            
                    ind1 = int(indices[count][0])
                    ind2 = int(indices[count][1])
                    ind3 = int(indices[count][2])
                    
                    out[int((count)*nmoov):int((count+1)*nmoov),ind1] = gain[0]
                    out[int((count)*nmoov):int((count+1)*nmoov),ind2] = gain[1]
                    out[int((count)*nmoov):int((count+1)*nmoov),ind3] = gain[2]
                    
            #out = out # enveloppe
            out = out*np.transpose(np.tile(data,(self.n,1))) # enveloppe*signal
              
        else: # one fixed angle 
            
            out[:,int(indices[0][0])] = data*Wsignal*gains[0][0] # 1
            out[:,int(indices[0][1])] = data*Wsignal*gains[0][1] # 2
            out[:,int(indices[0][2])] = data*Wsignal*gains[0][2] # 3
        
        out = out*self.scaling
        
        return out
    
    def play(self,out):
        """
        Play the mapped data with SoundCard module

        Parameters
        ----------
        out : Array
            Array with data mapped with NS_2d_mapping or NS_3d_mapping.

        """
        
        myrecording = self.SC.mesure(out) # PLAY