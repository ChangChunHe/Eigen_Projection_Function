#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 19:36:57 2018

@author: hecc
"""
import numpy as np
from pyniggli import reduced_cell

def get_EPA(d):
    n = np.shape(d)[0]
    eps = 1e-6
    eigval,eigvec = np.linalg.eigh(d)
    val_ind = np.argsort(eigval)
    eigval = eigval[val_ind]
    eigvec = eigvec[:,val_ind]
    unique_eigval = [[0]]
    count = 0
    for ii in range(n-1):
        if abs(eigval[ii]-eigval[ii+1])<eps:
            unique_eigval[count].append(ii+1)
        else:
            unique_eigval.append([ii+1])
            count += 1
    new_eigvec = np.zeros((n, len(unique_eigval)))
    for ii in range(len(unique_eigval)):
        new_eigvec[:,ii] = np.linalg.norm(eigvec[:,unique_eigval[ii]],axis=1)
    new_eigval = np.array([eigval[ii[0]] for ii in unique_eigval])
    return new_eigval,new_eigvec*new_eigvec


def d_EPF_atom(eigenvalue1,eigenvalue2,EPA1,EPA2):
    num1,num2=eigenvalue1.shape[0],eigenvalue2.shape[0]      
    d=index1=index2=0        
    epa1,epa2=EPA1[index1],EPA2[index2]        
    while True:
        if epa1>epa2:
            d+=abs(eigenvalue1[index1]-eigenvalue2[index2])*epa2    
            epa1-=epa2        
            index2+=1      
            if index2>=num2:
                break
            epa2=EPA2[index2]      
        elif epa2>epa1:
            d+=abs(eigenvalue1[index1]-eigenvalue2[index2])*epa1       
            epa2-=epa1       
            index1+=1       
            if index1>=num1:
                break
            epa1=EPA1[index1]       
        else:
            d+=abs(eigenvalue1[index1]-eigenvalue2[index2])*epa1
            index1+=1       
            index2+=1       
            if (index1>=num1)|(index2>=num2):
                break
            epa1=EPA1[index1]        
            epa2=EPA2[index2]
    return d


def get_reduced_cell(lattice, positions):
    niggli_reduction = reduced_cell(np.transpose(lattice))
    #Perform a Niggli Reduction to the original crystal
    lattice = np.transpose(niggli_reduction.niggli)
    #The Niggli Cell
    T = np.linalg.inv(np.transpose(niggli_reduction.C))
    #The transformation matrix
    positions=np.dot(positions, T)  
    positions=positions-np.floor(positions)
    return lattice, positions


def generate_all_basis(N1,N2,N3):
    n1,n2,n3 = 2*N1+1, 2*N2+1, 2*N3+1
    x = np.tile(np.arange(-N3,N3+1),n1*n2)
    y = np.tile(np.repeat(np.arange(-N2,N2+1),n3),n1)
    z = np.repeat(np.arange(-N1,N1+1),n2*n3)
    x,y,z = np.reshape(x,(-1,1)),np.reshape(y,(-1,1)),np.reshape(z,(-1,1))
    tmp = np.hstack((z,y))
    return np.hstack((tmp,x))














