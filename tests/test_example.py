#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 20:39:14 2018

@author: hecc
"""
import unittest
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../IO')
from vasp import read_vasp
from EPF import Structure, StructureDifference
class TestEPF(unittest.TestCase):
    
    def setUp(self):
        self.lattice = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.atoms = [6,1,1,1,1]
        self.positions1 = np.array([[0.5,0.5,0.5], [0,1,0],[1,0,0],[1,1,1],[0,0,1]])
        self.positions2 = np.array([[0.7,0.7,0.7], [0,1,0],[1,0,0],[1,1,1],[0,0,1]])
    
    def test_cart(self):
        S = read_vasp('primitive_cell_cart.vasp')
        S.draw_EPA()
        
    def test_direct(self):
        S = read_vasp('primitive_cell_direct.vasp')
        S.draw_EPA()
        
    def test_CH4(self):
        positions = self.positions1 * 1.09 *2/np.sqrt(3)
        S = Structure(self.lattice,positions,self.atoms,iscluster=True)
        equal_atoms = np.unique(S.get_equivalent_atoms())
        self.assertEqual(len(equal_atoms),2)
    
    def test_draw_EPF(self):
        from vasp import read_vasp
        S = read_vasp('primitive_cell_cart.vasp')
        S.draw_EPA()
        
    def test_CHH3(self):
        positions = self.positions2 * 1.09 * 2/np.sqrt(3)
        S = Structure(self.lattice,positions,self.atoms,iscluster=True)
        equal_atoms = np.unique(S.get_equivalent_atoms())
        self.assertEqual(len(equal_atoms),3)
        
        
    def test_StructureDifference(self):
        positions = self.positions1 * 1.09 *2/np.sqrt(3)
        S1 = Structure(self.lattice,positions,self.atoms,iscluster=True)
        positions = self.positions2 * 1.09 * 2/np.sqrt(3)
        S2 = Structure(self.lattice,positions,self.atoms,iscluster=True)
        SD = StructureDifference(S1,S2)
        diff,corresp = SD.get_structure_difference()
        self.assertAlmostEqual(diff, 0.7607215670542145)


    def test_graphene(self):
        S = read_vasp('primitive_cell_cart.vasp')
        equal_atoms = np.unique(S.get_equivalent_atoms())
        self.assertEqual(len(equal_atoms),1)
        
        
if __name__ == "__main__":
    import nose2
    nose2.main()