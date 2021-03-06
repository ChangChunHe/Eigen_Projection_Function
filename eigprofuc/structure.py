#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import eigprofuc.function_toolkit as ftk
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt

class Structure:


    def __init__(self, lattice, positions, atoms, iscluster=False):
        self.lattice = np.array(lattice).reshape((3,3))
        self.atoms = atoms
        self.positions = np.array(positions).reshape((-1,3))
        self.iscluster = iscluster
        self.atomsnumber = np.shape(self.positions)[0]


    def get_extend_min_distance_matrix(self):
        n = self.atomsnumber
        if self.iscluster:
            positions = np.dot(self.positions, self.lattice)
            d = ftk.get_dis_mat(positions)
        else:
            lattice, positions = ftk.get_reduced_cell(self.lattice, self.positions)
            positions = np.dot(positions, lattice)
            all_basis = ftk.generate_all_basis(1,1,1)
            d = np.zeros((n,n))
            for ii in range(n-1):
                dd = []
                for basis in all_basis:
                    tmp_pos = positions + np.sum(np.dot(np.diag(basis),lattice),axis=0)
                    dd.append(np.linalg.norm(positions[ii]-tmp_pos[ii+1:],axis=1))
                dd = np.array(dd).transpose()
                d[ii,ii+1:] = np.min(dd,axis=1)
            d = d + d.T
        return d + np.diag(self.atoms)


    def get_decrease_distance_matrix(self,Rcut=5, ispar=False):
        dec_func = lambda x: (1-x/(Rcut))**4 if x <= Rcut else 0
        if  self.iscluster:
            d = self.get_extend_min_distance_matrix()
            all_atom_d = []
            for ii in range(self.atomsnumber):
                q = np.where(d[ii]<Rcut)[0]
                tmp_d = np.zeros((len(q)+1,len(q)+1))
                tmp_d[0,1:] = [dec_func(d[ii,i]) for i in q]
                for jj in range(len(q)):
                    for kk in range(jj+1, len(q)):
                        if d[q[jj],q[kk]] > Rcut: continue
                        tmp_d[jj+1,kk+1] = dec_func(d[q[jj],q[kk]])
                diag_atom = [self.atoms[ii]] + [self.atoms[i] for i in q]
                all_atom_d.append(tmp_d + tmp_d.T +np.diag(diag_atom))
        else:
            lattice, positions = ftk.get_reduced_cell(self.lattice, self.positions)
            cart_pos = np.dot(positions,lattice)
            a,b,c = lattice[0,:],lattice[1,:],lattice[2,:]
            n0,n1,n2 = np.cross(b,c),np.cross(c,a),np.cross(a,b)
            n0,n1,n2 = n0/np.linalg.norm(n0),n1/np.linalg.norm(n1),n2/np.linalg.norm(n2)
            d0,d1,d2 = np.abs(np.dot(a,n0)),np.abs(np.dot(b,n1)),np.abs(np.dot(c,n2))
            N0,N1,N2 = int(np.ceil((Rcut+0.000001)/d0)),int(np.ceil((Rcut+0.000001)/d1)),int(np.ceil((Rcut+0.000001)/d2))
            all_basis = ftk.generate_all_basis(N0,N1,N2)
            all_atom_pos = [cart_pos + np.sum(np.dot(np.diag(basis),lattice),axis=0) for basis in all_basis]
            all_atom_pos = np.array(all_atom_pos).reshape((-1,3))
            all_atom_d = []
            for ii in range(self.atomsnumber):# for all atoms
                tmp_d = np.linalg.norm(cart_pos[ii]-all_atom_pos,axis=1)
                idx = np.where((tmp_d<=Rcut) &(tmp_d>0))[0]
                atom_d = ftk.get_dis_mat(all_atom_pos[idx])
                atom_d = np.vstack((tmp_d[idx],atom_d))
                atom_d = np.hstack((np.zeros((np.shape(atom_d)[0],1)),atom_d))
                atom_d[1:,0] = tmp_d[idx]
                vec_decr_fun = np.vectorize(dec_func)
                atom_d = vec_decr_fun(atom_d)
                atom_number =[self.atoms[ii]-1]+[self.atoms[np.mod(ii,self.atomsnumber)]-1 for ii in idx]
                all_atom_d.append(atom_d + np.diag(atom_number))
        return all_atom_d


    def get_structure_eig(self):
        return ftk.get_EPA(self.get_extend_min_distance_matrix())


    def get_atomsdistance_mindistance(self):
        eigval,eigvec = self.get_structure_eig()
        n = np.shape(self.positions)[0]
        d = np.zeros((n,n))
        for ii in range(n):
            for jj in range(ii+1,n):
                d[ii,jj] = ftk.d_EPF_atom(eigval,eigval, eigvec[ii],eigvec[jj])
        return d + d.T


    def get_atomsdistance_decdistance(self,Rcut=5,isparallel=False):
            all_atom_d = self.get_decrease_distance_matrix(Rcut=Rcut)
            eigval_list,eigvec_list = [],[]
            for ii in all_atom_d:
                eigval,eigvec = ftk.get_EPA(ii)
                eigval_list.append(eigval)
                eigvec_list.append(eigvec)
            n = np.shape(self.positions)[0]
            d = np.zeros((n,n))
            if not isparallel:
                for ii in range(n):
                    d[ii,ii+1:] = ftk.get_d_rows(eigval_list,eigvec_list,ii,n)
            elif isparallel:
                from multiprocessing import Pool, cpu_count
                p = Pool(cpu_count())
                for ii in range(n):
                    res = p.apply_async(ftk.get_d_rows, (eigval_list,eigvec_list,ii,n))
                    d[ii,ii+1:] = res.get()
            return d + d.T

    def get_equivalent_atoms(self,eps=1e-4):
        # only support minum distance matrix
        d = self.get_atomsdistance_mindistance()
        n = np.shape(self.positions)[0]
        equal_atom = np.arange(n)
        for ii in range(n):
            ind = np.where(d[ii]<eps)[0]
            equal_atom[ind] = ii
        _equal_atom = np.unique(equal_atom)
        equal_atom_set = dict()
        for ii, jj in enumerate(_equal_atom):
            equal_atom_set[ii] = np.where(equal_atom==jj)[0]
        return equal_atom_set


    def draw_EPA(self,eps=1e-4):
        eigval,eigvec = self.get_structure_eig()
        num_eigval = np.size(eigval)
        equal_atom = self.get_equivalent_atoms(eps)
        ax = plt.subplot(111)
        for idx, atom in equal_atom.items():
            tmp_eigvec = np.squeeze(eigvec[atom[0]])
            line = [[0,eigval[0]]]
            for ii in range(num_eigval-1):
                line.append([sum(tmp_eigvec[:ii+1]),eigval[ii]])
                line.append([sum(tmp_eigvec[:ii+1]),eigval[ii+1]])
            line.append([1,eigval[-1]])
            line = np.array(line)
            ax.plot(line[:,0],line[:,1],label=ftk.get_symbol(self.atoms[atom[0]]))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        plt.close()


    def draw_eig_spectra(self,atom_seq,sigma=0.1):
        # Lorentz expansion
        eigval,eigvec = self.get_structure_eig()
        min_eigval,max_eigval = 0, max(self.atoms)+1
        ax = plt.subplot(111)
        for i in atom_seq:
            f = lambda la: sum([eigvec[i,ii]*sigma/((la-eigval[ii])**2+sigma**2) for ii in range(np.shape(eigval)[0])])
            eigspec = [f(ii) for ii in np.arange(min_eigval,max_eigval,0.1)]
            ax.plot(np.arange(min_eigval,max_eigval,0.1), np.array(eigspec)+2*i,label=str(i),color='b')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        plt.close()



class StructureDifference:


    def __init__(self, structure1, structure2):
        self._structure1 = structure1
        self._structure2 = structure2


    def get_structure_mindisance_diff(self):
        eigval1,eigvec1 = self._structure1.get_structure_eig()
        eigval2,eigvec2 = self._structure2.get_structure_eig()
        n1,n2 = np.shape(eigvec1)[0],np.shape(eigvec2)[0]
        d = np.zeros((n1,n2))
        for ii in range(n1):
            for jj in range(n2):
                d[ii,jj]=ftk.d_EPF_atom(eigval1,eigval2,eigvec1[ii],eigvec2[jj])
        row_ind, col_ind = linear_sum_assignment(d)
        corresponding_sequence = np.array(list(zip(row_ind,col_ind)))
        return d[row_ind, col_ind].sum(), corresponding_sequence[np.argsort(corresponding_sequence[:,0])]


    def get_structure_decdistance_diff(self,Rcut=5):
        d1 = self._structure1.get_decrease_distance_matrix(Rcut=Rcut)
        d2 = self._structure2.get_decrease_distance_matrix(Rcut=Rcut)
        d = np.zeros((len(d1),len(d2)))
        for I in range(len(d1)):
            for J in range(len(d2)):
                eigval1,eigvec1 = ftk.get_EPA(d1[I])
                eigval2,eigvec2 = ftk.get_EPA(d2[J])
                n1,n2 = np.shape(eigvec1)[0],np.shape(eigvec2)[0]
                dd = np.zeros((n1,n2))
                for ii in range(n1):
                    for jj in range(n2):
                        dd[ii,jj]=ftk.d_EPF_atom(eigval1,eigval2,eigvec1[ii],eigvec2[jj])
                row_ind, col_ind = linear_sum_assignment(dd)
                d[I,J] = dd[row_ind, col_ind].sum()
        row_ind, col_ind = linear_sum_assignment(d)
        corresponding_sequence = np.array(list(zip(row_ind,col_ind)))
        return d[row_ind, col_ind].sum(),corresponding_sequence[np.argsort(corresponding_sequence[:,0])]

if __name__ == "__main__":
    from eigprofuc.io.vasp import read_vasp
    s = read_vasp('/home/hecc/Desktop/POSCAR14.vasp')
    import time
    a = time.time()
    for i in range(10):
        d1 = s.get_atomsdistance_decdistance(Rcut=3,isparallel=True)
    print(time.time()-a)
