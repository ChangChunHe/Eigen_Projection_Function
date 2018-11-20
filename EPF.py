#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:45:23 2018

@author: hecc
"""
import numpy as np
import function_toolkit as ftk
import hungarian
from matplotlib import pyplot as plt

class Structure:
    def __init__(self, lattice, positions, atoms, Rcut=5, iscluster=False):
        self._lattice = np.array(lattice).reshape((3,3))
        self._atoms = atoms
        self._positions = np.array(positions).reshape((-1,3))
        self._Rcut = Rcut
        self._iscluster = iscluster


    def get_extend_distance_matrix(self):
        n = np.shape(self._positions)[0]
        if self._iscluster:
            positions = np.dot(self._positions, self._lattice)
            d = np.zeros((n,n))
            for ii in range(n):
                for jj in range(ii+1,n):
                    d[ii,jj] = np.linalg.norm(positions[ii]-positions[jj])
        else:
            self._lattice, self._positions = ftk.get_reduced_cell(self._lattice, self._positions)
            positions = np.dot(self._positions, self._lattice)     
            all_basis = ftk.generate_all_basis(1,1,1)
            d = np.zeros((n,n))
            for ii in range(n-1):
                dd = []
                for basis in all_basis:
                    tmp_pos = positions + np.sum(np.dot(np.diag(basis),self._lattice),axis=0)
                    dd.append(np.linalg.norm(positions[ii]-tmp_pos[ii+1:],axis=1))
                dd = np.array(dd).transpose()
                d[ii,ii+1:] = np.min(dd,axis=1)
        return d + d.T + np.diag(self._atoms)


    def get_decrease_distance_matrix(self):
        if not self._iscluster:
            pass
        else:
            pass


    def get_structure_eig(self):
        return ftk.get_EPA(self.get_extend_distance_matrix())


    def get_atoms_EPF(self):
        eigval,eigvec = self.get_structure_eig()
        n = np.shape(self._positions)[0]
        d = np.zeros((n,n))
        for ii in range(n):
            for jj in range(ii+1,n):
                d[ii,jj] = ftk.d_EPF_atom(eigval,eigval, eigvec[ii],eigvec[jj])
        return d + d.T


    def get_equivalent_atoms(self,eps=1e-4):
        d = self.get_atoms_EPF()
        n = np.shape(self._positions)[0]
        equal_atom = np.arange(n)
        for ii in range(n):
            ind = np.where(d[ii]<eps)[0]
            equal_atom[ind] = ii
        return equal_atom


    def draw_EPA(self,isunique=True):
        eigval,eigvec = self.get_structure_eig()
        num_eigval = np.size(eigval)
        if isunique:
            equal_atom = np.unique(self.get_equivalent_atoms())
            for atom in equal_atom:
                tmp_eigvec = eigvec[atom]
                line = [[0,eigval[0]]]
                for ii in range(num_eigval-1):
                    line.append([sum(tmp_eigvec[:ii+1]),eigval[ii]])
                    line.append([sum(tmp_eigvec[:ii+1]),eigval[ii+1]])
                line.append([1,eigval[-1]])
                line = np.array(line)
                plt.plot(line[:,0],line[:,1],label=self._atoms[atom])
            plt.legend()
            plt.show()
        else:
            pass


    def get_eig_spetra(self):
        pass
    
class StructureDifference:
    
    def __init__(self, structure1,structure2,Rcut=5):
        self._structure1 = structure1
        self._structure2 = structure2
        self._Rcut=  Rcut

    def get_structure_difference(self):
        eigval1,eigvec1 = self._structure1.get_structure_eig()
        eigval2,eigvec2 = self._structure2.get_structure_eig()
        n1,n2 = np.shape(eigvec1)[0],np.shape(eigvec2)[0]
        d = np.zeros((n1,n2))
        for ii in range(n1):
            for jj in range(n2):
                d[ii,jj]=ftk.d_EPF_atom(eigval1,eigval2,eigvec1[ii],eigvec2[jj])
        h = hungarian.Hungarian(d)
        h.calculate()
        difference = h.get_total_potential()
        corresponding_sequence = h.get_results()
        return difference, corresponding_sequence


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    