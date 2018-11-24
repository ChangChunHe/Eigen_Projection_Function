#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np

from eigprofuc.IO.vasp import read_vasp
from eigprofuc.structure import Structure, StructureDifference


class TestEPF(unittest.TestCase):


    def setUp(self):
        self.lattice = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.atoms = [6,1,1,1,1]
        self.positions1 = np.array([[0.5,0.5,0.5], [0,1,0],[1,0,0],[1,1,1],[0,0,1]])
        self.positions2 = np.array([[0.7,0.7,0.7], [0,1,0],[1,0,0],[1,1,1],[0,0,1]])


    def test_cart(self):
        S = read_vasp('primitive_cell_cart.vasp')
        self.assertEqual(np.shape(S.positions)[0],8)


    def test_direct(self):
        S = read_vasp('primitive_cell_direct.vasp')
        self.assertEqual(np.shape(S.positions)[0],8)



    def test_CH4(self):
        positions = self.positions1 * 1.09 *2/np.sqrt(3)
        S = Structure(self.lattice,positions,self.atoms,iscluster=True)
        self.assertEqual(len(S.get_equivalent_atoms()),2)


    def test_draw_EPF(self):
        positions = self.positions2 * 1.09 * 2/np.sqrt(3)
        S = Structure(self.lattice,positions,self.atoms,iscluster=True)
        S.draw_EPA()
        
        
    def test_eigen_spectral(self):
        S = read_vasp('POSCAR_BN_12.vasp')
        S.draw_eig_spectra(atom_seq=range(12))

    def test_CHH3(self):
        positions = self.positions2 * 1.09 * 2/np.sqrt(3)
        S = Structure(self.lattice,positions,self.atoms,iscluster=True)
        self.assertEqual(len(S.get_equivalent_atoms()),3)


    def test_StructureDifference(self):
        positions = self.positions1 * 1.09 *2/np.sqrt(3)
        S1 = Structure(self.lattice,positions,self.atoms,iscluster=True)
        positions = self.positions2 * 1.09 * 2/np.sqrt(3)
        S2 = Structure(self.lattice,positions,self.atoms,iscluster=True)
        SD = StructureDifference(S1,S2)
        diff,corresp = SD.get_structure_mindisance_diff()
        self.assertAlmostEqual(diff, 0.7607215670542145)


    def test_graphene(self):
        S = read_vasp('primitive_cell_cart.vasp')
        self.assertEqual(len(S.get_equivalent_atoms()),1)
        

    def test_decrease_difference_cluster(self):
        S1,S2 = read_vasp('POSCAR1',iscluster=True),read_vasp('POSCAR2',iscluster=True)
        SD = StructureDifference(S1,S2)
        A, B = SD.get_structure_decdistance_diff(2)
        self.assertEqual(np.shape(B)[0],60)


    def test_decrease_difference_periodic(self):
        S1 = read_vasp('POSCAR_BN_2')
        S2 = read_vasp('POSCAR_BN_12.vasp')
        SD = StructureDifference(S1,S2)
        A,B = SD.get_structure_decdistance_diff(Rcut=2)
        self.assertLessEqual(A,1e-4)
        


if __name__ == "__main__":
    import nose2
    nose2.main()
