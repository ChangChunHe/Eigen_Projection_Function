#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 19:36:57 2018

@author: hecc
"""
import numpy as np
from eigprofuc.niggli import reduced_cell

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


def get_dis_mat(positions):
    n = np.shape(positions)[0]
    d = np.zeros((n,n))
    for ii in range(n):
        d[ii,ii+1:] = np.linalg.norm(positions[ii]-positions[ii+1:],axis=1)
    return d + d.T



periodic_table_dict = {'Vacc': 0,
                       'H': 1, 'He': 2,'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                       'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
                       'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
                       'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
                       'Cs': 55, 'Ba': 56, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
                       'Fr': 87, 'Ra': 88}




def get_symbol(atom):
    """
    get_symbol return symbol of atomic number

    parameter:

    atom: int, atomic number.
    """
    for key, value in periodic_table_dict.items():
        if atom == value:
            return str(key)
    return "NaN_x"


def symbol2number(symbol):
    return periodic_table_dict[symbol]







