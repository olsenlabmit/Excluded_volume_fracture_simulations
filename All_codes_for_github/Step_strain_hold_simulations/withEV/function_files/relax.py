#!/use/local/bin/env python
# -*- coding: utf-8 -*-
##
##-------------------------------------------------
## Fast Inertial Relaxation Engine (FIRE) Optimizer
## Ref: Bitzek et al, PRL, 97, 170201 (2006)
##
## Author: Akash Arora
## Mixed bond update: Devosmita Sen - March 2023
## Excluded volume term update: Devosmita Sen - November 2024
## Implementation is inspired from LAMMPS and ASE Master Code
##-------------------------------------------------
import math
import time
import random
import numpy as np
import scipy.optimize as opt
from numpy import linalg as LA
from scipy.optimize import fsolve
import os.path
import param as p
from scipy.special import erf
import ioLAMMPS

import os
os.environ['NUMBA_NUM_THREADS'] = str(p.numba_num_threads)##'2'  # Example: Set to 2 threads
####import numba
####print(numba.config.NUMBA_NUM_THREADS)


from numba import njit,jit, prange
import functools


class Optimizer(object):

    def __init__(self, atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, r0, parameters, epsilon, 
                 ftype):

        self.atoms = atoms
        self.bonds = bonds
        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
        self.zlo = zlo
        self.zhi = zhi
        self.parameters = parameters
        self.r0 = r0
        ##        self.N = N
        self.ftype = ftype
##        self.conn_arr = conn
        self.epsilon = epsilon
##        self.sigma_EV = sigma_EV
##        self.U0_EV = U0_EV

        ##        self.chains_conn_to_crosslink = (np.ones((len(self.atoms[:, 0]), 4)) * (-1)).astype('int')

        ##        for idx in range(0, len(self.bonds[:, 0])):
        ##            [lnk_1, lnk_2] = self.bonds[idx, 2:4]
        ##            a = np.where(self.chains_conn_to_crosslink[lnk_1 - 1, :] == -1)[0][0]  # first element where it is -1
        ##            self.chains_conn_to_crosslink[lnk_1 - 1, a] = idx
        ##            a = np.where(self.chains_conn_to_crosslink[lnk_2 - 1, :] == -1)[0][0]  # first element where it is -1
        ##            self.chains_conn_to_crosslink[lnk_2 - 1, a] = idx
        
        num_grid = 100
        self.num_grid = num_grid
        N = p.N
        rc = 4 / np.sqrt(3 / (2 * N))
        self.rc = rc
        del_x = rc / num_grid
        self.del_x = del_x
        force_factor = np.zeros((num_grid, num_grid, num_grid))
        '''

        for x_idx in range(0, num_grid):
            for y_idx in range(0, num_grid):
                for z_idx in range(0, num_grid):
                    delr_norm2 = (x_idx ** 2 + y_idx ** 2 + z_idx ** 2)*del_x**2
                    force_factor[x_idx, y_idx, z_idx] = np.exp(
                        -3 * delr_norm2 / (2 * N))  # not including b2 because it is 1
        force_factor = epsilon*(3 / N) * ((3 / (2 * np.pi * N))**1.5) * force_factor
        
        print('generated_force_factor_table')
        '''

        self.force_factor = force_factor

        pre_factor=epsilon * (3) * ((3 / (2 * np.pi * N)) ** 1.5)
        self.pre_factor=pre_factor

    @staticmethod
    @njit(parallel=True)
    def bondlengths(atoms, bonds, xlo,xhi,ylo,yhi,zlo,zhi):

        #atoms = self.atoms
        #bonds = self.bonds
####        Lx = self.xhi - self.xlo
####        Ly = self.yhi - self.ylo
####        Lz = self.zhi - self.zlo

        Lx = xhi - xlo
        Ly = yhi - ylo
        Lz = zhi - zlo
        
        n_atoms = len(atoms[:, 0])
        n_bonds = len(bonds[:, 0])

        dist = np.zeros((n_bonds, 4), dtype=float)

        for i in prange(0, n_bonds):
            lnk_1 = bonds[i, 2] - 1
            lnk_2 = bonds[i, 3] - 1
            delr = atoms[lnk_1, :] - atoms[lnk_2, :]

            delr[0] = delr[0] - int(round(delr[0] / Lx)) * Lx
            delr[1] = delr[1] - int(round(delr[1] / Ly)) * Ly
            delr[2] = delr[2] - int(round(delr[2] / Lz)) * Lz

            dist[i, 0:3] = delr
            dist[i, 3] = LA.norm(delr)

        return dist
    
    @staticmethod
    @jit(nopython=True) 
    def invlangevin(x):
        return x * (2.99942 - 2.57332 * x + 0.654805 * x ** 2) / (1 - 0.894936 * x - 0.105064 * x ** 2)

    def kuhn_stretch(self, lam, E_b):

        def func(x, lam, E_b):
            y = lam / x
            beta = self.invlangevin(y)
            return E_b * np.log(x) - lam * beta / x

        if lam == 0:
            return 1
        else:
            lam_b = opt.root_scalar(func, args=(lam, E_b), bracket=[lam, lam + 1], x0=lam + 0.05)
            return lam_b.root

    def get_bondforce(self, r, i):
        bonds = self.bonds
        ctype = bonds[i, 0]
        parameters = self.parameters
        N = parameters[ctype, 0]  # bonds[i,0] gives the ctype
        b = parameters[ctype, 1]
        K = parameters[ctype, 2]

        ##        print(parameters)
        ##        stop
        fit_param = parameters[ctype, 3]
        E_b = parameters[ctype, 4]
        ##        K  = self.K
        r0 = self.r0
        ##        Nb = self.N # b = 1 (lenght scale of the system)

        ##        E_b = 1200

        x = (r - r0) / (N * b)
        ##        print('x ',x)
        if (x < 0.90):
            ##           print('get_bondforce, case 1')
            lam_b = 1.0
            fbkT = self.invlangevin(x)
            fbond = -K * fbkT / r
        elif (x < 1.4):
            ##           print('get_bondforce, case 2')
            lam_b = self.kuhn_stretch(x, E_b)
            fbkT = self.invlangevin(x / lam_b) / lam_b
            fbond = -K * fbkT / r
        else:
            ##           print('get_bondforce, case 3')
            lam_b = x + 0.05
            fbkT = 325 + 400 * (x - 1.4)
            fbond = -K * fbkT / r

        return fbond, lam_b

    


####    def process_element(i_idx, j_idx):
####    # Your computation here
####        return result

####    @staticmethod
####    @njit##(parallel=True)  # Set "nopython" mode for best performance, equivalent to @njit
    def get_force_EV_vectorized(self,largest_cc, atoms,n_atoms,rc,del_x,force_factor, epsilon, Lx,Ly,Lz, factor,Lx0,delx, pre_factor,N):  #self.bonds, self.atoms, n_atoms, rc, self.del_x, self.force_factor, self.epsilon, Lx, Ly, Lz,wy_arr,wz_arr)
####        print((largest_cc))
        num_atoms=len(largest_cc) ## actual number of atoms over which force should be calculated
        #print(n_atoms)
        points=atoms[largest_cc,:]
        delr = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        delr[...,0] = delr[...,0] - np.round(delr[...,0] / Lx) * Lx
        delr[...,1] = delr[...,1] - np.round(delr[...,1] / Ly) * Ly
        delr[...,2] = delr[...,2] - np.round(delr[...,2] / Lz) * Lz
        
        
        rmid = 0.5*(points[:, np.newaxis, :] + points[np.newaxis, :, :])
        rmid[...,0] = rmid[...,0] - np.round(rmid[...,0] / Lx) * Lx

        Rex2=N
        # Calculate the index xidx based on rmid[:, :, 0]
        xidx = np.floor(rmid[..., 0] / delx).astype(int)##(rmid[..., 0] / delx).astype(int)
####        print(xidx)
####        stop
####        xidx=np.zeros(len(rmid[...,0]),dtype='int')
####        xidx = np.array([int(val) for val in xidx.flatten()])
        Rey2 = N*factor[xidx]
####        Rey2 = Rey2_temp.reshape((num_atoms, num_atoms))
        Rez2=Rey2
        
####        scaled_squares = (delr[..., 0]**2 / Rex2) + (delr[..., 1]**2 / Rey2) + (delr[..., 2]**2 / Rez2)
####        result_matrix = np.exp(-1.5 * scaled_squares)


        
        #print(np.shape(fij))

        # Initialize F and e
        F_vectorized = np.zeros((n_atoms, 3))
        e_vectorized = 0##np.sum(np.triu(result_matrix, k=1))*(1/3)

        
        upper_triangle_mask = np.triu(np.ones((num_atoms, num_atoms)), k=1)  # Mask for i < j
        within_cutoff = np.all(np.abs(delr) < rc, axis=-1)

####        within_cutoff = np.zeros((num_atoms, num_atoms), dtype=np.bool_)
####
####        for i in range(num_atoms):
####            for j in range(i + 1, num_atoms):
####                if np.abs(delr[i, j, 0]) < rc and np.abs(delr[i, j, 1]) < rc and np.abs(delr[i, j, 2]) < rc:
####                    within_cutoff[i, j] = True
####                    within_cutoff[j, i] = True  # Since it's symmetric

                
        result_matrix = np.zeros_like(delr[..., 0])
####        scaled_squares = (delr[..., 0]**2 / Rex2) + (delr[..., 1]**2 / Rey2) + (delr[..., 2]**2 / Rez2)
        scaled_squares = ((delr[..., 0][within_cutoff])**2 / Rex2) + ((delr[..., 1][within_cutoff])**2 / Rey2) + ((delr[..., 2][within_cutoff])**2 / Rez2)
        result_matrix[within_cutoff] = np.exp(-1.5 * scaled_squares[within_cutoff])
        # Apply the mask to set values for i >= j to zero
        result_matrix = result_matrix * upper_triangle_mask  # Set lower part to 0

        # Compute fij for each pair (i, j)
        fij = result_matrix[..., np.newaxis] * np.stack([delr[..., 0] / Rex2, delr[..., 1] / Rey2, delr[..., 2] / Rez2], axis=-1)

        # Extract indices of the upper triangular part (excluding diagonal)
        i_upper, j_upper = np.triu_indices(num_atoms, k=1)


        # Apply the mask to set values for i >= j to zero
        ##result_matrix = result_matrix * upper_triangle_mask  # Set lower part to 0


        # Update F
        np.add.at(F_vectorized, largest_cc[i_upper], fij[i_upper, j_upper])
        np.subtract.at(F_vectorized, largest_cc[j_upper], fij[i_upper, j_upper])

        e_vectorized = np.sum(result_matrix)*(1/3)

        F_vectorized=F_vectorized*pre_factor
        e_vectorized=e_vectorized*pre_factor

####        print(np.where(F_vectorized==0))


        return F_vectorized, e_vectorized

        
        
    
    @staticmethod
    @njit(parallel=True)  # Set "nopython" mode for best performance, equivalent to @njit
    ##@functools.lru_cache()
    def get_force_EV( largest_cc, atoms,n_atoms,rc,del_x,force_factor, epsilon, Lx,Ly,Lz, factor,Lx0,delx, pre_factor,N):  #self.bonds, self.atoms, n_atoms, rc, self.del_x, self.force_factor, self.epsilon, Lx, Ly, Lz,wy_arr,wz_arr)
####        print('Hello 0')
####        largest_cc = np.unique(bonds[:, 2:4])
        f_EV= np.zeros((n_atoms, 3))
        e_EV=0
        ##N=p.N
        ##pre_factor=epsilon * (3) * ((3 / (2 * np.pi * N)) ** 1.5)
        ##wyz_arr_avg2=(wy_arr2+wz_arr2)
        ##Lyz0_avg2=(Ly0+Lz0)**2
        ####Ly0=self.Ly0
        ####Lz0=self.Lz0
        ####delx=self.delx
        ##print('ok 1')

####        sigma = np.zeros((n_atoms, 6), dtype=float)
        fij_EV_arr=np.zeros((n_atoms, n_atoms,3))
        for i_idx in prange(0, len(largest_cc)):  # range(0,n_atoms): ## elements in largest_cc are the atom numbers (indices)- hence, no need to subtract 1 
            i = largest_cc[i_idx]
            ri = atoms[i, :]
            for j_idx in range(i_idx + 1, len(largest_cc)):  # range(0,n_atoms):
                j = largest_cc[j_idx]

                # if (i != j and ([i, j] not in pairs_considered) and ([j, i] not in pairs_considered)):
                #pairs_considered.append([i, j])
                rj = atoms[j, :]
                delr = ri - rj
####                print(Lx,Ly,Lz)
####                print('Hello 1')

                delr[0] = delr[0] - int(np.round(delr[0] / Lx)) * Lx
####                print('Hello 2')
                delr[1] = delr[1] - int(np.round(delr[1] / Ly)) * Ly
                delr[2] = delr[2] - int(np.round(delr[2] / Lz)) * Lz

                

                

                if (abs(delr[0]) < rc and abs(delr[1]) < rc and abs(delr[2]) < rc):

                    rmid=(ri+rj)*0.5

                    rmid[0] = rmid[0] - int(np.round(rmid[0] / Lx)) * Lx
                    #rmid[1] = rmid[1] - int(round(rmid[1] / Ly)) * Ly  # no need to do this, because I don't need the y and z coordinates
                    #rmid[2] = rmid[2] - int(round(rmid[2] / Lz)) * Lz

                    xidx=int(rmid[0]/delx)

                    
                   

                    ####delr_norm2 = (delr[0]** 2 + delr[1] ** 2 + delr[2] ** 2)  ##(x_idx ** 2 + y_idx ** 2 + z_idx ** 2) * del_x ** 2

####                    x_factor=1##Lx/Lx0
####                    y_factor=
####                    z_factor=
                    
                    Rex2=N
                    Rey2=N*factor[xidx] ##(wyz_arr_avg2[xidx]/Lyz0_avg2)##**2#*(wyz_arr_avg[xidx]/Lyz0_avg)
                    Rez2=Rey2##N*(wyz_arr_avg2[xidx]/Lyz0_avg2)##**2#*(wyz_arr_avg[xidx]/Lyz0_avg)
                    
                    force_factor1= pre_factor*np.exp(-(3/2) *((delr[0]** 2)/Rex2+(delr[1]** 2)/Rey2+(delr[2]** 2)/Rez2))##np.exp(-3 * delr_norm2 / (2 * N))  # not including b2 because it is 1
                    e_EV=e_EV+force_factor1*(1/3)
                    
                    fij = force_factor1* np.array([delr[0]/Rex2,delr[1]/Rey2,delr[2]/Rez2])  # scale the EV parameter according to the actual width along each of the y and z axes

                    ####fij = force_factor1* delr

                    
                    
                    f_EV[i, :] = f_EV[i, :] + fij
                    fij_EV_arr[i,j,:]=fij_EV_arr[i,j,:]-fij
                    #f_EV[j, :] = f_EV[j, :] - fij

        f_EV=f_EV+np.sum(fij_EV_arr, axis=0)
                    
        return f_EV,e_EV




    @staticmethod
    @njit##(parallel=True)  # Set "nopython" mode for best performance, equivalent to @njit
##    @functools.lru_cache()
    def get_force_EV_improved_bad( largest_cc, atoms,n_atoms,rc,del_x,force_factor, epsilon, Lx,Ly,Lz, factor,Lx0,delx, pre_factor,N):  #self.bonds, self.atoms, n_atoms, rc, self.del_x, self.force_factor, self.epsilon, Lx, Ly, Lz,wy_arr,wz_arr)
####        print('Hello 0')
####        largest_cc = np.unique(bonds[:, 2:4])
        f_EV= np.zeros((n_atoms, 3))
        e_EV=0

        points=atoms[largest_cc,:]

        delr = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        delr[...,0] = delr[...,0] - np.round(delr[...,0] / Lx) * Lx
        delr[...,1] = delr[...,1] - np.round(delr[...,1] / Ly) * Ly
        delr[...,2] = delr[...,2] - np.round(delr[...,2] / Lz) * Lz
        
        
        rmid = 0.5*(points[:, np.newaxis, :] + points[np.newaxis, :, :])
        rmid[...,0] = rmid[...,0] - np.round(rmid[...,0] / Lx) * Lx

        Rex2=N

        
        ##N=p.N
        ##pre_factor=epsilon * (3) * ((3 / (2 * np.pi * N)) ** 1.5)
        ##wyz_arr_avg2=(wy_arr2+wz_arr2)
        ##Lyz0_avg2=(Ly0+Lz0)**2
        ####Ly0=self.Ly0
        ####Lz0=self.Lz0
        ####delx=self.delx
        ##print('ok 1')

####        sigma = np.zeros((n_atoms, 6), dtype=float)
        
        for i_idx in range(0, len(largest_cc)):  # range(0,n_atoms): ## elements in largest_cc are the atom numbers (indices)- hence, no need to subtract 1 
            i = largest_cc[i_idx]
            ri = atoms[i, :]
            for j_idx in range(i_idx + 1, len(largest_cc)):  # range(0,n_atoms):
                j = largest_cc[j_idx]

                # if (i != j and ([i, j] not in pairs_considered) and ([j, i] not in pairs_considered)):
                #pairs_considered.append([i, j])
########                rj = atoms[j, :]
########                delr = ri - rj
############                print(Lx,Ly,Lz)
############                print('Hello 1')
########
########                delr[0] = delr[0] - int(np.round(delr[0] / Lx)) * Lx
############                print('Hello 2')
########                delr[1] = delr[1] - int(np.round(delr[1] / Ly)) * Ly
########                delr[2] = delr[2] - int(np.round(delr[2] / Lz)) * Lz

                delr0=delr[i_idx,j_idx,0]
                delr1=delr[i_idx,j_idx,1]
                delr2=delr[i_idx,j_idx,2]

                

                

                if (abs(delr0) < rc and abs(delr1) < rc and abs(delr2) < rc):

########                    rmid=(ri+rj)*0.5
########
########                    rmid[0] = rmid[0] - int(np.round(rmid[0] / Lx)) * Lx
                    #rmid[1] = rmid[1] - int(round(rmid[1] / Ly)) * Ly  # no need to do this, because I don't need the y and z coordinates
                    #rmid[2] = rmid[2] - int(round(rmid[2] / Lz)) * Lz

                    

                    xidx=int(rmid[i_idx,j_idx,0]/delx)

                    
                   

                    ####delr_norm2 = (delr[0]** 2 + delr[1] ** 2 + delr[2] ** 2)  ##(x_idx ** 2 + y_idx ** 2 + z_idx ** 2) * del_x ** 2

####                    x_factor=1##Lx/Lx0
####                    y_factor=
####                    z_factor=
                    
####                    Rex2=N
                    Rey2=N*factor[xidx] ##(wyz_arr_avg2[xidx]/Lyz0_avg2)##**2#*(wyz_arr_avg[xidx]/Lyz0_avg)
                    Rez2=Rey2##N*(wyz_arr_avg2[xidx]/Lyz0_avg2)##**2#*(wyz_arr_avg[xidx]/Lyz0_avg)
                    
                    force_factor1= pre_factor*np.exp(-(3/2) *((delr0** 2)/Rex2+(delr1** 2)/Rey2+(delr2** 2)/Rez2))##np.exp(-3 * delr_norm2 / (2 * N))  # not including b2 because it is 1
                    e_EV=e_EV+force_factor1*(1/3)
                    
                    fij = force_factor1* np.array([delr0/Rex2,delr1/Rey2,delr2/Rez2])  # scale the EV parameter according to the actual width along each of the y and z axes

                    ####fij = force_factor1* delr

                    
                    
                    f_EV[i, :] = f_EV[i, :] + fij
                    f_EV[j, :] = f_EV[j, :] - fij


                    
        return f_EV,e_EV

####    @staticmethod
####    @jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
####    def get_norm(delr):
####        delr_norm=LA.norm(delr)
####        return delr_norm

####    @staticmethod
####    @jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
####    def get_vdot(x,y):
####        vdot_x=np.dot(x.flatten(),y.flatten())
####        return vdot_x

####    @staticmethod
####    @jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
####    def get_sqrt(x):
####        sqrt_x=math.sqrt(x)
####        return sqrt_x

####    @staticmethod
####    @jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
####    def get_npsqrt(x):
####        sqrt_x=np.sqrt(x)
####        return sqrt_x

####    @staticmethod
####    @jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
####    def get_npfloor(x):
####        floor_x=np.floor(x)
####        return floor_x

    def get_force(self, EV_bool,factor,largest_cc):

        ##        c3=(72/2**(1/3))*self.epsilon/self.sigma_EV**2
        ##        a=2**(1/6)*self.sigma_EV
        ##        U0=c3
        ##        U1=-2*c3*a
        ##        U2=c3*a**2
        ##
        ##        N = self.N
        ##        E_b = 1200
        atoms = self.atoms
        bonds = self.bonds
        ftype = self.ftype
        Lx = self.xhi - self.xlo
        Ly = self.yhi - self.ylo
        Lz = self.zhi - self.zlo
        n_atoms = len(atoms[:, 0])
        n_bonds = len(bonds[:, 0])

        e = 0.0
        Gamma = 0.0
        f = np.zeros((n_atoms, 3), dtype=float)

####        Lyz0_avg2=(Ly+Lz)**2

        # print('Langvin part started')

        ##delr2_arr = np.zeros((n_bonds, 1))
        ##delr_norm_arr = np.zeros((n_bonds, 1))
        ##delr_arr = np.zeros((n_bonds, 3))

        for i in range(0, n_bonds):
            ##            print('bond_number',i)

            ctype = bonds[i, 0]
            parameters = self.parameters
            N = parameters[ctype, 0]  # bonds[i,0] gives the ctype
            N = int(N)
            b = parameters[ctype, 1]
            K = parameters[ctype, 2]
            fit_param = parameters[ctype, 3]
            E_b = parameters[ctype, 4]

            lnk_1 = bonds[i, 2] - 1
            lnk_2 = bonds[i, 3] - 1
            delr = atoms[lnk_1, :] - atoms[lnk_2, :]
            ##            print('lnk_1',lnk_1,'lnk_2',lnk_2)
            ##            print('delr ',delr, '\n atoms[lnk_1,:]',atoms[lnk_1,:], '\n  atoms[lnk_2,:]', atoms[lnk_2,:])
            delr[0] = delr[0] - int(round(delr[0] / Lx)) * Lx
            delr[1] = delr[1] - int(round(delr[1] / Ly)) * Ly
            delr[2] = delr[2] - int(round(delr[2] / Lz)) * Lz

            ##            A2k=np.zeros((n_bonds, N,3)) # stores A2k values , 3 dimensions, for each m, and for each bond
            ##            A3k=np.zeros((n_bonds, N,3))
            ##            f1=np.zeros((n_bonds, N)) # A2x^2+A2y^2+A2z^2
            ##            f2=np.zeros((n_bonds, N))

            r = LA.norm(delr)##LA.norm(delr)
            if (r > 0):
                ##               print('i',i)
                [fbond, lam_b] = self.get_bondforce(r, i)
                lam = (r - self.r0) / N
                beta = -fbond * r / K * lam_b  # fbond*r/K is fbKT
                e_bond = N * 0.5 * E_b * math.log(lam_b) ** 2
                ##               print('fbond',fbond,"  ",'r',r)

                e_stretch = N * ((lam / lam_b) * beta + math.log(beta / math.sinh(beta)))
                e = e + e_bond + e_stretch
                ##               if(EV_bool==True):
                ##                   r1=atoms[lnk_1,:]
                ##                   r2=atoms[lnk_2,:]

                ##                   for m in range(1,N):
                ##                       A2k[i,m,:]=(3/2)*(r1/m+r2/(N-m))# omitting b2 here
                ##                       A3k[i,m,:]=(3/2)*(r1**2/m+r2**2/(N-m))# omitting b2 here
                ##
                ##                       f1[i,m]=A2k[i,m,0]**2+A2k[i,m,1]**2+A2k[i,m,2]**2
                ##                       f2[i,m]=A3k[i,m,0]+A3k[i,m,1]+A3k[i,m,2]
                ##delr2_arr[i] = r ** 2
                ##delr_arr[i, :] = delr  # =lnk_1-lnk_2
                ##delr_norm_arr[i] = r

            else:
                fbond = 0.0
                e = e + 0.0
            ##            stop
            Gamma = Gamma + r * r

            # get the entire force vector
            # f_EV[lnk_2]=f_EV[lnk_2]+get_force_EV(sigma, epsilon,[U0,U1,U2],lnk_2)

            # apply force to each of 2 atoms
            if (lnk_1 < n_atoms):
                f[lnk_1, 0] = f[lnk_1, 0] + delr[0] * fbond
                f[lnk_1, 1] = f[lnk_1, 1] + delr[1] * fbond
                f[lnk_1, 2] = f[lnk_1, 2] + delr[2] * fbond

            if (lnk_2 < n_atoms):
                f[lnk_2, 0] = f[lnk_2, 0] - delr[0] * fbond
                f[lnk_2, 1] = f[lnk_2, 1] - delr[1] * fbond
                f[lnk_2, 2] = f[lnk_2, 2] - delr[2] * fbond

        # print('Langvin part completed')
        

        # print('EV_ part started')
        if (EV_bool == True):
            #f_EV = np.zeros((n_atoms, 3))
            ##pairs_considered = []
            ##pairs_considered_within_cutoff = []
            rc = self.rc  ##0.1 / np.sqrt(3 / (2 * N))  # cutoff radius
####            epsilon = 5.0

            ##            sumE=np.zeros((n_atoms,3)) # has the capacity to store the sum for all atoms
            # in case an atom is not in the connected network, then the sum is zero, similar to the Langevin part

            # according to the summation, there would have been 3 sums- one over all the bonds, one over the bonds connected to k and an outer loop for the atoms k
            # here, i consider only 2 loops- one for the bonds for which the k's are considred, and the inner one is all the bonds


            [f_EV,e_EV]=self.get_force_EV(largest_cc, self.atoms, n_atoms, rc, self.del_x, self.force_factor, self.epsilon, Lx, Ly,Lz,factor,self.Lx0,self.delx, self.pre_factor,N)##get_force_EV(atoms_list, self.atoms,self.rc,self.del_x)
            
####            [f_EV_vect,e_EV_vect]=self.get_force_EV_improved(largest_cc, self.atoms, n_atoms, rc, self.del_x, self.force_factor, self.epsilon, Lx, Ly,Lz,factor,self.Lx0,self.delx, self.pre_factor,N)##get_force_EV(atoms_list, self.atoms,self.rc,self.del_x)

####            print('e_EV',e_EV,'e_EV_vect',e_EV_vect)
####            print('f_EV',np.shape(f_EV))
####            print('f_EV_vect',np.shape(f_EV_vect))
####            print(np.allclose(f_EV, f_EV_vect, rtol=1e-4))
####            
####            print(np.isclose(e_EV, e_EV_vect, rtol=1e-5))

####            if(np.allclose(f_EV, f_EV_vect, atol=1e-5) ==False or np.isclose(e_EV, e_EV_vect, atol=1e-5)==False):
######                print(
####                stop
############            stop


##largest_cc, atoms,n_atoms,rc,del_x,force_factor, epsilon, Lx,Ly,Lz, wyz_arr_avg2,Lx0,Lyz0_avg2,delx, pre_factor,N)



                            # if (fij > 0.001 * epsilon):
                            #   pairs_considered_within_cutoff.append([i, j])

        ##        f_EV=sumE*(3/(2*np.pi))**3 # didn't include b2 since it is 1
        # print('EV iteration completed')

        #if (EV_bool == True):
            f = f + f_EV
            e=e+e_EV

        return f, e, Gamma

    @staticmethod
    @jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
    def get_width_init(nx_num, bonds, atoms, delx, largest_cc):  # init width of network after force relaxation
        
####        largest_cc=np.unique(bonds[:, 2:4])-1 ## testing

        max_num=1000
        wy_arr=np.zeros(nx_num)
        wz_arr=np.zeros(nx_num)

        y_min_arr=np.ones(nx_num)*max_num ## this should be populated by a very high number
        y_max_arr=np.zeros(nx_num)

        z_min_arr=np.ones(nx_num)*max_num ## this should be populated by a very high number
        z_max_arr=np.zeros(nx_num)

        #xidx_arr=[]

        atoms_list_idx = largest_cc ##np.unique(bonds[:, 2:4])
        #atoms_list_idx=atoms_list-1 ## indices of only the atoms that are part of the network

        
        for i in atoms_list_idx:
            x_pos=atoms[i,0]
            y_pos=atoms[i,1]
            z_pos=atoms[i,2]
            xidx=int(x_pos/delx) # this means that x_pos is between xidx*delx and xidx*delx+delx
            #xidx_arr.append(xidx)
            
            y_min_arr[xidx]=min(y_min_arr[xidx],y_pos)
            y_max_arr[xidx]=max(y_max_arr[xidx],y_pos)

            z_min_arr[xidx]=min(z_min_arr[xidx],z_pos)
            z_max_arr[xidx]=max(z_max_arr[xidx],z_pos)

        wy_arr=y_max_arr-y_min_arr
        wz_arr=z_max_arr-z_min_arr

        for xidx in range(0,nx_num):
         if(wy_arr[xidx]==0 or wy_arr[xidx]==-max_num):
             wy_arr[xidx]=wy_arr[xidx-1] ## wy_arr[xidx-1] must have been non zero, otherwise it would have been cauht at an earlier iteration
         if(wz_arr[xidx]==0 or wz_arr[xidx]==-max_num):
             wz_arr[xidx]=wz_arr[xidx-1]

        #xidx_arr=np.unique(np.array(xidx_arr))
       
        

        return wy_arr,wz_arr


    @staticmethod
    @jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
    def get_atom_pos_PBC(atoms, xlo,ylo,zlo,Lx,Ly,Lz):
        atoms[:, 0] = atoms[:, 0] - np.floor((atoms[:, 0] - xlo) / Lx) * Lx
        atoms[:, 1] = atoms[:, 1] - np.floor((atoms[:, 1] - ylo) / Ly) * Ly
        atoms[:, 2] = atoms[:, 2] - np.floor((atoms[:, 2] - zlo) / Lz) * Lz

        return atoms

    @staticmethod
    @jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
    def get_width(nx_num, bonds, atoms, delx, largest_cc,Lyz0_avg2):

####        largest_cc=np.unique(bonds[:, 2:4])-1## testing
        
        max_num=1000
        wy_arr=np.zeros(nx_num)
        wz_arr=np.zeros(nx_num)

        y_min_arr=np.ones(nx_num)*max_num ## this should be populated by a very high number
        y_max_arr=np.zeros(nx_num)

        z_min_arr=np.ones(nx_num)*max_num ## this should be populated by a very high number
        z_max_arr=np.zeros(nx_num)

        #xidx_arr=[]

        atoms_list_idx = largest_cc ##np.unique(bonds[:, 2:4])
        #atoms_list_idx=atoms_list-1 ## indices of only the atoms that are part of the network

        
        for i in atoms_list_idx:
            x_pos=atoms[i,0]
            y_pos=atoms[i,1]
            z_pos=atoms[i,2]
            xidx=int(x_pos/delx) # this means that x_pos is between xidx*delx and xidx*delx+delx
            #xidx_arr.append(xidx)
            
            y_min_arr[xidx]=min(y_min_arr[xidx],y_pos)
            y_max_arr[xidx]=max(y_max_arr[xidx],y_pos)

            z_min_arr[xidx]=min(z_min_arr[xidx],z_pos)
            z_max_arr[xidx]=max(z_max_arr[xidx],z_pos)

        wy_arr=y_max_arr-y_min_arr
        wz_arr=z_max_arr-z_min_arr

        for xidx in range(0,nx_num):
         if(wy_arr[xidx]==0 or wy_arr[xidx]==-max_num):
             wy_arr[xidx]=wy_arr[xidx-1] ## wy_arr[xidx-1] must have been non zero, otherwise it would have been cauht at an earlier iteration
         if(wz_arr[xidx]==0 or wz_arr[xidx]==-max_num):
             wz_arr[xidx]=wz_arr[xidx-1]


        wyz_arr_avg2=np.square(wy_arr+wz_arr)
        factor=wyz_arr_avg2/Lyz0_avg2
        #xidx_arr=np.unique(np.array(xidx_arr))

        return factor##wy_arr2,wz_arr2

    
      
    def fire_iterate(self, ftol, maxiter, write_itr, EV_bool,init_test_PBC, factor,largest_cc,logfilename):
####        if(EV_bool==True):
####            ftol=0.05

        same_tol_thresh=p.same_tol_thresh #1e-5  # the threshold for saying that two tol values are same - is difference<threshold, then equal
        num_same_tol_thresh=p.num_same_tol_thresh  ## 500  # number of iterations for which the tolerance has to be the same to be considered convergence
        itr_cnt_same_tol=0  # number of iterations for which the tolerance has remained the same (ie. below a certain threshold)

        tol_min=p.tol_min
        max_itr_for_conv_crit=p.max_itr_for_conv_crit

        
        if(init_test_PBC==True):
            write_itr=1000

        tstart = time.time()

        ## Optimization parameters:
        eps_energy = 1.0e-8
        delaystep = 5
        dt_grow = 1.1
        dt_shrink = 0.5
        alpha0 = 0.1
        alpha_shrink = 0.99
        tmax = 10.0
        maxmove = 0.1
        last_negative = 0

        dt = 0.002
        dtmax = dt * tmax
        alpha = alpha0
        last_negative = 0

        Lx = self.xhi - self.xlo
        Ly = self.yhi - self.ylo
        Lz = self.zhi - self.zlo
        n_atoms = len(self.atoms[:, 0])
        n_bonds = len(self.bonds[:, 0])
        v = np.zeros((n_atoms, 3), dtype=float)

        n_bonds = len(self.bonds)
        dist = np.zeros((n_bonds, 4), dtype=float)

        [f, e, Gamma] = self.get_force(EV_bool,factor,largest_cc)
        ##dist = self.bondlengths(self.atoms, self.bonds, self.xlo,self.xhi,self.ylo,self.yhi,self.zlo,self.zhi)

        fmaxitr = np.max(np.max(np.absolute(f)))
####        print(np.shape(f))
####        print((np.vdot(f,f)))
####        print(self.get_vdot(f, f))
####        stop
        fnormitr = math.sqrt(np.vdot(f, f))
        ##        logfile = open(logfilename,'w')
        ##        logfile.write('FIRE: iter  Energy  fmax  fnorm  avg(r)/Nb  max(r)/Nb\n')
        ##        logfile.write('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f\n' %
        ##                              ('FIRE', 0, e, fmaxitr, fnormitr, np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
        ##        logfile.flush()
        '''
        print('FIRE: iter  Energy  fmax  fnorm  ')
        print('%s: %5d  %9.6f  %9.6f  %9.6f' %
              ('FIRE', 0, e, fmaxitr,
               fnormitr))  # , avg(r)/Nb  max(r)/Nb np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
        '''

        for itr in range(0, maxiter):
            # print('FIRE itr=', itr)

            vdotf = np.vdot(v, f)
            if (vdotf > 0.0):
                vdotv = np.vdot(v, v)
                fdotf = np.vdot(f, f)
                scale1 = 1.0 - alpha
                if (fdotf == 0.0):
                    scale2 = 0.0
                else:
                    scale2 = alpha * math.sqrt(vdotv / fdotf)
                v = scale1 * v + scale2 * f

                if (itr - last_negative > delaystep):
                    dt = min(dt * dt_grow, dtmax)
                    alpha = alpha * alpha_shrink

            else:
                last_negative = itr
                dt = dt * dt_shrink
                alpha = alpha0
                v[:] = v[:] * 0.0

            v = v + dt * f
            dr = dt * v
            normdr = np.sqrt(np.vdot(dr, dr))
            if (normdr > maxmove):
                dr = maxmove * dr / normdr

            self.atoms = self.atoms + dr

            
####            self.atoms[:, 0] = self.atoms[:, 0] - np.floor((self.atoms[:, 0] - self.xlo) / Lx) * Lx
####            self.atoms[:, 1] = self.atoms[:, 1] - np.floor((self.atoms[:, 1] - self.ylo) / Ly) * Ly
####            self.atoms[:, 2] = self.atoms[:, 2] - np.floor((self.atoms[:, 2] - self.zlo) / Lz) * Lz
            self.atoms=self.get_atom_pos_PBC(self.atoms, self.xlo,self.ylo,self.zlo,Lx,Ly,Lz)

            
######            for i in range(0, n_atoms):
######                self.atoms[i, 0] = self.atoms[i, 0] - math.floor((self.atoms[i, 0] - self.xlo) / Lx) * Lx
######                self.atoms[i, 1] = self.atoms[i, 1] - math.floor((self.atoms[i, 1] - self.ylo) / Ly) * Ly
######                self.atoms[i, 2] = self.atoms[i, 2] - math.floor((self.atoms[i, 2] - self.zlo) / Lz) * Lz

            [f, e, Gamma] = self.get_force(EV_bool,factor,largest_cc)
            fmaxitr = np.max(np.max(np.absolute(f)))
            fnormitr = math.sqrt(np.vdot(f, f))

            if ((itr + 1) % write_itr == 0):
####                dist = self.bondlengths(self.atoms, self.bonds, self.xlo,self.xhi,self.ylo,self.yhi,self.zlo,self.zhi)
                ##             logfile.write('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f\n' %
                ##                                  ('FIRE', itr+1, e, fmaxitr, fnormitr, np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
                ##             logfile.flush()

                # Print on screen
                '''
                print('%s: %5d  %9.6f  %9.6f  %9.6f' %
                      ('FIRE', itr + 1, e, fmaxitr,
                       fnormitr))  # ,  np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
                '''
                
                if (EV_bool == True and init_test_PBC==True):
                    filename = 'restart_network_with_EV_itr_' + str(itr) + '.txt'
                    frac_weak = 0
                    directory = './' + str(int(100 * frac_weak)) + '/'
                    file_path = os.path.join(directory, filename)
                    if not os.path.isdir(directory):
                        os.mkdir(directory)
                    ioLAMMPS.writeLAMMPS(file_path, self.xlo, self.xhi, self.xlo, self.xhi, self.xlo,
                                         self.xhi, self.atoms, self.bonds, 2, 2, [1, 1], [])

            # Checking for convergence
            '''
            if (fnormitr < ftol):
                dist = self.bondlengths()
                tend = time.time()
                ##             logfile.write('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f\n' %
                ##                                  ('FIRE', itr+1, e, fmaxitr, fnormitr, np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
                ##             logfile.flush()
                print('%s: %5d  %9.6f  %9.6f  %9.6f' %
                      (
                          'FIRE', itr + 1, e, fmaxitr,
                          fnormitr))  # , np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
                print('Iterations converged, Time taken: %7.4f' % (tend - tstart))
                break
            '''

            # Checking for convergence- main convergence criterion
            if ((fnormitr < ftol) or (itr_cnt_same_tol>=num_same_tol_thresh and fnormitr<tol_min)):  # FIRE converged using main convergence criterion
                ##print('itr_cnt_same_tol',itr_cnt_same_tol)
                ##print('FIRE converged using main convergence criterion')
##                print('num_same_tol_thresh',num_same_tol_thresh)
                ##print(itr_cnt_same_tol>=num_same_tol_thresh)
                ##dist = self.bondlengths(self.atoms, self.bonds, self.xlo,self.xhi,self.ylo,self.yhi,self.zlo,self.zhi)
                tend = time.time()
                ##             logfile.write('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f\n' %
                ##                                  ('FIRE', itr+1, e, fmaxitr, fnormitr, np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
                ##             logfile.flush()
####                print('%s: %5d  %9.6f  %9.6f  %9.6f' %
####                      (
####                          'FIRE', itr + 1, e, fmaxitr,
####                          fnormitr))  # , np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))

                #print('FIRE: iter  Energy  fmax  fnorm f_EV_norm f_elastic_norm ')
####                print('%s: %5d  %9.6f  %9.6f  %9.6f %9.6f %9.6f' %
####                ('FIRE', itr+1, e, fmaxitr,
####                fnormitr,fEV_normitr,felastic_normitr))
                '''
                print('%s: %5d  %9.6f  %9.6f  %9.6f' %
                      ('FIRE', itr + 1, e, fmaxitr,
                       fnormitr))
                
                
                print('Iterations converged, Time taken: %7.4f' % (tend - tstart))
                '''
                break
            
            elif(itr>max_itr_for_conv_crit and fnormitr<tol_min):  # FIRE converged using secondary convergence criterion
                ##print('itr_cnt_same_tol',itr_cnt_same_tol)
                ##print('FIRE converged using secondary convergence criterion')
##                print('num_same_tol_thresh',num_same_tol_thresh)
                ##print(itr_cnt_same_tol>=num_same_tol_thresh)
                ##dist = self.bondlengths(self.atoms, self.bonds, self.xlo,self.xhi,self.ylo,self.yhi,self.zlo,self.zhi)
                tend = time.time()
                ##             logfile.write('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f\n' %
                ##                                  ('FIRE', itr+1, e, fmaxitr, fnormitr, np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
                ##             logfile.flush()
####                print('%s: %5d  %9.6f  %9.6f  %9.6f' %
####                      (
####                          'FIRE', itr + 1, e, fmaxitr,
####                          fnormitr))  # , np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))

                #print('FIRE: iter  Energy  fmax  fnorm f_EV_norm f_elastic_norm ')
####                print('%s: %5d  %9.6f  %9.6f  %9.6f %9.6f %9.6f' %
####                ('FIRE', itr+1, e, fmaxitr,
####                fnormitr,fEV_normitr,felastic_normitr))

                ##print('%s: %5d  %9.6f  %9.6f  %9.6f' %('FIRE', itr + 1, e, fmaxitr,fnormitr))
                
                ##print('Iterations converged, Time taken: %7.4f' % (tend - tstart))
                break
            
            elif (itr == maxiter - 1):
                print('Maximum iterations reached')

        ##        logfile.close()

        return e, Gamma

    def compute_pressure(self, inv_volume):

        ##        K = self.K
        bonds = self.bonds
        ##        ctype=bonds[i,0]
        ##        parameters=self.parameters
        ##        N=parameters[ctype,0]#bonds[i,0] gives the ctype
        ##        b=parameters[ctype, 1]
        ##        K=parameters[ctype, 2]
        ##        fit_param=parameters[ctype, 3]
        ##        E_b=parameters[ctype,4]
        r0 = self.r0
        ftype = self.ftype
        Lx = self.xhi - self.xlo
        Ly = self.yhi - self.ylo
        Lz = self.zhi - self.zlo
        atoms = self.atoms
        bonds = self.bonds
        n_atoms = len(atoms[:, 0])
        n_bonds = len(bonds[:, 0])

        pxx = pyy = pzz = pxy = pyz = pzx = 0.0
        sigma = np.zeros((n_atoms, 6), dtype=float)
##        inv_volume = 1.0 / (Lx * Ly * Lz)
        for i in range(0, n_bonds):

            lnk_1 = bonds[i, 2] - 1
            lnk_2 = bonds[i, 3] - 1
            delr = atoms[lnk_1, :] - atoms[lnk_2, :]

            delr[0] = delr[0] - int(round(delr[0] / Lx)) * Lx
            delr[1] = delr[1] - int(round(delr[1] / Ly)) * Ly
            delr[2] = delr[2] - int(round(delr[2] / Lz)) * Lz

            r = LA.norm(delr)##LA.norm(delr)
            if (r > 0.0):
                if (ftype == 'Mao'):
                    [fbond, lam_b] = self.get_bondforce(r, i)
                else:
                    fbond = self.get_bondforce(r)
            else:
                fbond = 0.0

            # apply pressure to each of the 2 atoms
            # And for each of the 6 components
            if (lnk_1 < n_atoms):
                sigma[lnk_1, 0] = sigma[lnk_1, 0] + 0.5 * delr[0] * delr[0] * fbond
                sigma[lnk_1, 1] = sigma[lnk_1, 1] + 0.5 * delr[1] * delr[1] * fbond
                sigma[lnk_1, 2] = sigma[lnk_1, 2] + 0.5 * delr[2] * delr[2] * fbond
                sigma[lnk_1, 3] = sigma[lnk_1, 3] + 0.5 * delr[0] * delr[1] * fbond
                sigma[lnk_1, 4] = sigma[lnk_1, 4] + 0.5 * delr[1] * delr[2] * fbond
                sigma[lnk_1, 5] = sigma[lnk_1, 5] + 0.5 * delr[2] * delr[0] * fbond

            if (lnk_2 < n_atoms):
                sigma[lnk_2, 0] = sigma[lnk_2, 0] + 0.5 * delr[0] * delr[0] * fbond
                sigma[lnk_2, 1] = sigma[lnk_2, 1] + 0.5 * delr[1] * delr[1] * fbond
                sigma[lnk_2, 2] = sigma[lnk_2, 2] + 0.5 * delr[2] * delr[2] * fbond
                sigma[lnk_2, 3] = sigma[lnk_2, 3] + 0.5 * delr[0] * delr[1] * fbond
                sigma[lnk_2, 4] = sigma[lnk_2, 4] + 0.5 * delr[1] * delr[2] * fbond
                sigma[lnk_2, 5] = sigma[lnk_2, 5] + 0.5 * delr[2] * delr[0] * fbond

        pxx = np.sum(sigma[:, 0]) * inv_volume
        pyy = np.sum(sigma[:, 1]) * inv_volume
        pzz = np.sum(sigma[:, 2]) * inv_volume
        pxy = np.sum(sigma[:, 3]) * inv_volume
        pyz = np.sum(sigma[:, 4]) * inv_volume
        pzx = np.sum(sigma[:, 5]) * inv_volume

        return pxx, pyy, pzz, pxy, pyz, pzx

##    def compute_pressure()
    @staticmethod
    @njit#(parallel=True)  # Set "nopython" mode for best performance, equivalent to @njit
    def compute_pressure_EV( largest_cc, atoms,n_atoms,rc,del_x,force_factor, epsilon, Lx,Ly,Lz, factor,Lx0,delx, pre_factor,N,inv_volume):

        sigma = np.zeros((n_atoms, 6), dtype=float)
        for i_idx in range(0, len(largest_cc)):  # range(0,n_atoms): ## elements in largest_cc are the atom numbers (indices)- hence, no need to subtract 1 
            i = largest_cc[i_idx]
            ri = atoms[i, :]
            for j_idx in range(i_idx + 1, len(largest_cc)):  # range(0,n_atoms):
                j = largest_cc[j_idx]

                # if (i != j and ([i, j] not in pairs_considered) and ([j, i] not in pairs_considered)):
                #pairs_considered.append([i, j])
                rj = atoms[j, :]
                delr = ri - rj
####                print(Lx,Ly,Lz)
####                print('Hello 1')

                delr[0] = delr[0] - int(np.round(delr[0] / Lx)) * Lx
####                print('Hello 2')
                delr[1] = delr[1] - int(np.round(delr[1] / Ly)) * Ly
                delr[2] = delr[2] - int(np.round(delr[2] / Lz)) * Lz

                

                

                if (abs(delr[0]) < rc and abs(delr[1]) < rc and abs(delr[2]) < rc):

                    rmid=(ri+rj)*0.5

                    rmid[0] = rmid[0] - int(np.round(rmid[0] / Lx)) * Lx
                    #rmid[1] = rmid[1] - int(round(rmid[1] / Ly)) * Ly  # no need to do this, because I don't need the y and z coordinates
                    #rmid[2] = rmid[2] - int(round(rmid[2] / Lz)) * Lz

                    xidx=int(rmid[0]/delx)

                    
                   

                    ####delr_norm2 = (delr[0]** 2 + delr[1] ** 2 + delr[2] ** 2)  ##(x_idx ** 2 + y_idx ** 2 + z_idx ** 2) * del_x ** 2

####                    x_factor=1##Lx/Lx0
####                    y_factor=
####                    z_factor=
                    
                    Rex2=N
                    Rey2=N*factor[xidx] ##(wyz_arr_avg2[xidx]/Lyz0_avg2)##**2#*(wyz_arr_avg[xidx]/Lyz0_avg)
                    Rez2=Rey2##N*(wyz_arr_avg2[xidx]/Lyz0_avg2)##**2#*(wyz_arr_avg[xidx]/Lyz0_avg)
                    
                    force_factor1= pre_factor*np.exp(-(3/2) *((delr[0]** 2)/Rex2+(delr[1]** 2)/Rey2+(delr[2]** 2)/Rez2))##np.exp(-3 * delr_norm2 / (2 * N))  # not including b2 because it is 1
                    ##e_EV=e_EV+force_factor1*(1/3)
                    
                    fij = force_factor1* np.array([delr[0]/Rex2,delr[1]/Rey2,delr[2]/Rez2])  # scale the EV parameter according to the actual width along each of the y and z axes


                    
        
                    sigma[i-1, 0] = sigma[i-1, 0] + 0.5 * delr[0] * fij[0]
                    sigma[i-1, 1] = sigma[i-1, 1] + 0.5 * delr[1] * fij[1]
                    sigma[i-1, 2] = sigma[i-1, 2] + 0.5 * delr[2] * fij[2]
                    sigma[i-1, 3] = sigma[i-1, 3] + 0.5 * delr[0] * fij[1]
                    sigma[i-1, 4] = sigma[i-1, 4] + 0.5 * delr[1] * fij[2]
                    sigma[i-1, 5] = sigma[i-1, 5] + 0.5 * delr[2] * fij[0]
##                    sigma[i-1, 0] = sigma[i-1, 0] + 0.5 * delr[0] * fij[0]

                    sigma[j-1, 0] = sigma[j-1, 0] + 0.5 * delr[0] * fij[0]
                    sigma[j-1, 1] = sigma[j-1, 1] + 0.5 * delr[1] * fij[1]
                    sigma[j-1, 2] = sigma[j-1, 2] + 0.5 * delr[2] * fij[2]
                    sigma[j-1, 3] = sigma[j-1, 3] + 0.5 * delr[0] * fij[1]
                    sigma[j-1, 4] = sigma[j-1, 4] + 0.5 * delr[1] * fij[2]
                    sigma[j-1, 5] = sigma[j-1, 5] + 0.5 * delr[2] * fij[0]
                    

                    #pairs_considered_within_cutoff.append([i, j])
                    
        pxx = np.sum(sigma[:, 0]) * inv_volume
        pyy = np.sum(sigma[:, 1]) * inv_volume
        pzz = np.sum(sigma[:, 2]) * inv_volume
        pxy = np.sum(sigma[:, 3]) * inv_volume
        pyz = np.sum(sigma[:, 4]) * inv_volume
        pzx = np.sum(sigma[:, 5]) * inv_volume
        
        return pxx, pyy, pzz, pxy, pyz, pzx


    
    def change_box(self, scale_x, scale_y, scale_z):

        xlo = self.xlo
        xhi = self.xhi
        ylo = self.ylo
        yhi = self.yhi
        zlo = self.zlo
        zhi = self.zhi
        atoms = self.atoms
        bonds = self.bonds
        n_atoms = len(atoms[:, 0])
        n_bonds = len(bonds[:, 0])

        xmid = (xlo + xhi) / 2
        ymid = (ylo + yhi) / 2
        zmid = (zlo + zhi) / 2

        new_xlo = xmid + scale_x * (xlo - xmid)
        new_ylo = ymid + scale_y * (ylo - ymid)
        new_zlo = zmid + scale_z * (zlo - zmid)

        new_xhi = xmid + scale_x * (xhi - xmid)
        new_yhi = ymid + scale_y * (yhi - ymid)
        new_zhi = zmid + scale_z * (zhi - zmid)

        newLx = new_xhi - new_xlo
        newLy = new_yhi - new_ylo
        newLz = new_zhi - new_zlo
        for i in range(0, n_atoms):
            atoms[i, 0] = xmid + scale_x * (atoms[i, 0] - xmid)
            atoms[i, 1] = ymid + scale_y * (atoms[i, 1] - ymid)
            atoms[i, 2] = zmid + scale_z * (atoms[i, 2] - zmid)

        self.atoms = atoms
        self.xlo = new_xlo
        self.xhi = new_xhi
        self.ylo = new_ylo
        self.yhi = new_yhi
        self.zlo = new_zlo
        self.zhi = new_zhi

    def KMCbondbreak(self, tau, delta_t, pflag, index, frac_weak):

        # Material parameters:
        # beta = 1.0 -- All material params, U0 and sigma, are in units of kT.
        # Main array: Bonds_register = [Activity index, type, index, link1, link2, dist, rate(ri)]
        # All are active at the start (active = 1, break = 0)

        def get_link_bonds(link, bonds_register):

            conn = {}
            a1 = np.where(bonds_register[:, 3] == link)
            a2 = np.where(bonds_register[:, 4] == link)
            a = np.concatenate((a1[0], a2[0]))
            a = np.unique(a)
            for i in range(0, len(a)):
                if (bonds_register[a[i], 0] == 1):
                    conn.update({a[i]: bonds_register[a[i], 5]})

            conn = dict(sorted(conn.items(), key=lambda x: x[1]))

            return conn

        ftype = self.ftype
        n_bonds = len(self.bonds[:, 0])
        bonds_register = np.zeros((n_bonds, 7))
        bonds_register[:, 0] = 1
        bonds_register[:, 1:5] = self.bonds
        dist = self.bondlengths(self.atoms, self.bonds, self.xlo,self.xhi,self.ylo,self.yhi,self.zlo,self.zhi)
        bonds_register[:, 5] = dist[:, 3]

        step = 10
        # File to write bond broken stats
######        if (index % step == 0):
######            directory = './' + str(int(100 * frac_weak)) + '/'
######            filename = 'bondbroken_%d.txt' % (index)
######            file_path = os.path.join(directory, filename)
######            if not os.path.isdir(directory):
######                os.mkdir(directory)
######            f2 = open(file_path, 'w')
######            f2.write(
######                '#type, atom1, atom2, length, rate(v), t, t_KMC, vmax, active bonds, num_weak_bond_broken, num_strong_bond_broken\n')

            # Write probability values in a file (at every KMC call)
        if (pflag == 1):
            prob_file = 'prob_%d.txt' % (index)
            directory = './' + str(int(100 * frac_weak)) + '/'
            filename = prob_file
            file_path = os.path.join(directory, filename)
            if not os.path.isdir(directory):
                os.mkdir(directory)
            fl1 = open(file_path, 'w')

        for i in range(0, n_bonds):
            bonds = self.bonds
            ctype = bonds[i, 0]
            parameters = self.parameters
            N = parameters[ctype, 0]  # bonds[i,0] gives the ctype
            b = parameters[ctype, 1]
            K = parameters[ctype, 2]
            fit_param = parameters[ctype, 3]
            E_b = parameters[ctype, 4]
            U0_kT = parameters[ctype, 5]
            Nb = N * b
            r = bonds_register[i, 5]
            if (r > 0):
                [fbond, lam_b] = self.get_bondforce(r, i)
            else:
                fbond = 0.0

            ##            fit_param = 1
            fbkT = -fbond * r / K
            bonds_register[i, 6] = math.exp(-U0_kT + fbkT * fit_param)
            if (pflag == 1): fl1.write('%i %i %i %i %i %6.4f %6.4f\n' % (bonds_register[i, 0],
                                                                         bonds_register[i, 1], bonds_register[i, 2],
                                                                         bonds_register[i, 3],
                                                                         bonds_register[i, 4], bonds_register[i, 5],
                                                                         bonds_register[i, 6]))

        if (pflag == 1): fl1.close()

        active_bonds = np.where(bonds_register[:, 0] == 1)
        n_bonds_init = len(active_bonds[0])
        vmax = max(bonds_register[active_bonds[0], 6])  # same as rmax in paper
        if (vmax == 0): vmax = 1e-12
        # if fbkT = 0, vmax = exp(-56). This number below the machine precison.
        # hence, we assign a small detectable number, vmax = 10^{-12}.
        # Essentially, it implies that bond breaking rate is very low, or
        # t = 1/(vmax*nbonds) is very high compare to del_t and hence it will not
        # enter the KMC bond breaking loop

        t = 1 / (vmax * len(active_bonds[0]))

        bond_broken = False

        weak_bond_broken = 0  # number of weak (and strong) bonds broken
        strong_bond_broken = 0
        ##print('KMC statistics:')
        ##print('Max rate, Active bonds, and t_KMC = %6.4E, %5d, %6.4E' % (vmax, len(active_bonds[0]), t))
##        print('outside t=',t)

        ## when no bonds are being broken, the strains keep increasing, and since the rates become very high, t_KMC becomes extremely small, and thus it takes a lot of iterations to break the loop
        # instead, we can just set the t value to be equal to delta_t
        if (t < delta_t):
##            t = 0
####            t = delta_t

            ##'''
            while (t < delta_t):
####                print('inside KMC t=',t)
                ##                sys.pause()
                t_KMC = 1 / (vmax * len(active_bonds[0]))
                vmax = max(bonds_register[active_bonds[0], 6])
                ##                random.seed(10)
                bond_index = random.randint(0, len(active_bonds[0]) - 1)
                pot_bond = active_bonds[0][bond_index]
                ##                random.seed(10)
                rnd_num = random.uniform(0, 1)


                t = t + t_KMC # only increment time, do not break any bonds


                # in this step- do not break bonds #####- because we are simulating solely the effect of network stretch on chain breaking in cont sim part
                if ((bonds_register[
                         pot_bond, 6] / vmax) > rnd_num):  # bonds_register[pot_bond,6] is chain scission rate r

                    
                    bonds_register[pot_bond, 0] = 0  # Bond is broken!
                    bond_broken = True

                    ##                   print('Bond is broken')
                    ##                   sys.pause()
                    if (bonds_register[pot_bond, 1] == 0):  # weak bond has broken
                        weak_bond_broken = weak_bond_broken + 1
                    elif (bonds_register[pot_bond, 1] == 1):  # strong bond has broken
                        strong_bond_broken = strong_bond_broken + 1
                

                        

                    t = t + t_KMC
######                    if (index % step == 0):
######                        f2.write('%5d  %5d  %5d  %0.4E  %0.4E  %0.4E  %0.4E  %0.4E  %5d  %5d  %5d\n' % (
######                            bonds_register[pot_bond, 2], bonds_register[pot_bond, 3],
######                            bonds_register[pot_bond, 4], bonds_register[pot_bond, 5], bonds_register[pot_bond, 6],
######                            t, t_KMC, vmax, len(active_bonds[0]), weak_bond_broken, strong_bond_broken))
######                        f2.flush()
                    # Local Relaxation -- If the bond-broken created a dangling end system
                    # then make the force on the remaining fourth bond

                    
                    link_1 = bonds_register[pot_bond, 3]
                    conn = get_link_bonds(link_1, bonds_register)

                    if (len(conn) == 3):
                        if (conn[list(conn)[0]] == 0 and conn[list(conn)[1]] == 0):
                            bonds_register[list(conn)[2], 6] = 0

                    elif (len(conn) == 2):
                        if (conn[list(conn)[0]] == 0):
                            bonds_register[list(conn)[1], 6] = 0

                    elif(len(conn)==1):
                        bonds_register[list(conn)[0], 6] = 0

                    link_2 = bonds_register[pot_bond, 4]
                    conn = get_link_bonds(link_2, bonds_register)
                    if (len(conn) == 3):
                        if (conn[list(conn)[0]] == 0 and conn[list(conn)[1]] == 0):
                            bonds_register[list(conn)[2], 6] = 0

                    elif (len(conn) == 2):
                        if (conn[list(conn)[0]] == 0):
                            bonds_register[list(conn)[1], 6] = 0

                    elif (len(conn) == 1):
                        bonds_register[list(conn)[0], 6] = 0

##                    temp1 = np.where(self.conn_arr[int(link_1) - 1] == link_2 - 1)[0]
##                    self.conn_arr[int(link_1) - 1, temp1] = -1
##                    temp2 = np.where(self.conn_arr[int(link_2) - 1] == link_1 - 1)[0]
##                    self.conn_arr[int(link_2) - 1, temp2] = -1
                    



                else:
                    t = t + t_KMC

                
                
                # active bond- means that it is not broken!!
                active_bonds = np.where(bonds_register[:, 0] == 1)  # all active bonds
        ##                active_bonds_weak = np.where(bonds_register[:,0]==1 and bonds_register[:,1]==0) # active bond and weak
        ##                active_bonds_strong = np.where(bonds_register[:,0]==1 and bonds_register[:,1]==1) # active bond and strong

            ##'''  # no need to update active_bonds because nothing is breaking 

######        if (index % step == 0): f2.close()

        n_bonds_final = len(active_bonds[0])
        if (n_bonds_final < n_bonds_init):
            bonds_final = np.zeros((n_bonds_final, 4), dtype=int)
            bonds_final[:, 0:4] = bonds_register[active_bonds[0], 1:5].astype(int)
            self.bonds = bonds_final

        ##print('time, init bonds, final bonds = %6.4E, %5d, %5d' % (t, n_bonds_init, n_bonds_final))
        ##print('---------------------------------------------------------------')

        ##        self.chains_conn_to_crosslink = (np.ones((len(self.atoms[:, 0]), 4)) * (-1)).astype('int')

        # reassess which chains are connected to which crosslinkers- only if chains break, not otherwise
        ##        for idx in range(0, len(self.bonds[:, 0])):
        ##            [lnk_1, lnk_2] = self.bonds[idx, 2:4]
        ##            a = np.where(self.chains_conn_to_crosslink[lnk_1 - 1, :] == -1)[0][0]  # first element where it is -1
        ##            self.chains_conn_to_crosslink[lnk_1 - 1, a] = idx
        ##            a = np.where(self.chains_conn_to_crosslink[lnk_2 - 1, :] == -1)[0][0]  # first element where it is -1
        ##            self.chains_conn_to_crosslink[lnk_2 - 1, a] = idx

        return t, n_bonds_init, n_bonds_final, weak_bond_broken, strong_bond_broken


    def KMCbondbreak_nobreak(self, tau, delta_t, pflag, index, frac_weak):

        # Material parameters:
        # beta = 1.0 -- All material params, U0 and sigma, are in units of kT.
        # Main array: Bonds_register = [Activity index, type, index, link1, link2, dist, rate(ri)]
        # All are active at the start (active = 1, break = 0)

        def get_link_bonds(link, bonds_register):

            conn = {}
            a1 = np.where(bonds_register[:, 3] == link)
            a2 = np.where(bonds_register[:, 4] == link)
            a = np.concatenate((a1[0], a2[0]))
            a = np.unique(a)
            for i in range(0, len(a)):
                if (bonds_register[a[i], 0] == 1):
                    conn.update({a[i]: bonds_register[a[i], 5]})

            conn = dict(sorted(conn.items(), key=lambda x: x[1]))

            return conn

        ftype = self.ftype
        n_bonds = len(self.bonds[:, 0])
        bonds_register = np.zeros((n_bonds, 7))
        bonds_register[:, 0] = 1
        bonds_register[:, 1:5] = self.bonds
        dist = self.bondlengths(self.atoms, self.bonds, self.xlo,self.xhi,self.ylo,self.yhi,self.zlo,self.zhi)
        bonds_register[:, 5] = dist[:, 3]

        step = 10
        # File to write bond broken stats
######        if (index % step == 0):
######            directory = './' + str(int(100 * frac_weak)) + '/'
######            filename = 'bondbroken_%d.txt' % (index)
######            file_path = os.path.join(directory, filename)
######            if not os.path.isdir(directory):
######                os.mkdir(directory)
######            f2 = open(file_path, 'w')
######            f2.write(
######                '#type, atom1, atom2, length, rate(v), t, t_KMC, vmax, active bonds, num_weak_bond_broken, num_strong_bond_broken\n')

            # Write probability values in a file (at every KMC call)
        if (pflag == 1):
            prob_file = 'prob_%d.txt' % (index)
            directory = './' + str(int(100 * frac_weak)) + '/'
            filename = prob_file
            file_path = os.path.join(directory, filename)
            if not os.path.isdir(directory):
                os.mkdir(directory)
            fl1 = open(file_path, 'w')
        bonds = self.bonds
        ctype = 1
        parameters = self.parameters
        N = parameters[ctype, 0]  # bonds[i,0] gives the ctype
        b = parameters[ctype, 1]
        K = parameters[ctype, 2]
        fit_param = parameters[ctype, 3]
        E_b = parameters[ctype, 4]
        U0_kT = parameters[ctype, 5]
        Nb = N * b
        for i in range(0, n_bonds):
            
            r = bonds_register[i, 5]
            if (r > 0):
                [fbond, lam_b] = self.get_bondforce(r, i)
            else:
                fbond = 0.0

            ##            fit_param = 1
            fbkT = -fbond * r / K
            bonds_register[i, 6] = math.exp(-U0_kT + fbkT * fit_param)
            if (pflag == 1): fl1.write('%i %i %i %i %i %6.4f %6.4f\n' % (bonds_register[i, 0],
                                                                         bonds_register[i, 1], bonds_register[i, 2],
                                                                         bonds_register[i, 3],
                                                                         bonds_register[i, 4], bonds_register[i, 5],
                                                                         bonds_register[i, 6]))

        if (pflag == 1): fl1.close()

        active_bonds = np.where(bonds_register[:, 0] == 1)
        n_bonds_init = len(active_bonds[0])
        vmax = max(bonds_register[active_bonds[0], 6])  # same as rmax in paper
        if (vmax == 0): vmax = 1e-12
        # if fbkT = 0, vmax = exp(-56). This number below the machine precison.
        # hence, we assign a small detectable number, vmax = 10^{-12}.
        # Essentially, it implies that bond breaking rate is very low, or
        # t = 1/(vmax*nbonds) is very high compare to del_t and hence it will not
        # enter the KMC bond breaking loop

        t = 1 / (vmax * len(active_bonds[0]))

        bond_broken = False

        weak_bond_broken = 0  # number of weak (and strong) bonds broken
        strong_bond_broken = 0
        #print('KMC statistics:')
        #print('Max rate, Active bonds, and t_KMC = %6.4E, %5d, %6.4E' % (vmax, len(active_bonds[0]), t))
##        print('outside t=',t)

        ## when no bonds are being broken, the strains keep increasing, and since the rates become very high, t_KMC becomes extremely small, and thus it takes a lot of iterations to break the loop
        # instead, we can just set the t value to be equal to delta_t
        if (t < delta_t):
##            t = 0
            t = delta_t

            '''
            while (t < delta_t):
####                print('inside KMC t=',t)
                ##                sys.pause()
                t_KMC = 1 / (vmax * len(active_bonds[0]))
                vmax = max(bonds_register[active_bonds[0], 6])
                ##                random.seed(10)
                bond_index = random.randint(0, len(active_bonds[0]) - 1)
                pot_bond = active_bonds[0][bond_index]
                ##                random.seed(10)
                rnd_num = random.uniform(0, 1)


                t = t + t_KMC # only increment time, do not break any bonds


                # in this step- do not break bonds #####- because we are simulating solely the effect of network stretch on chain breaking in cont sim part
                if ((bonds_register[
                         pot_bond, 6] / vmax) > rnd_num):  # bonds_register[pot_bond,6] is chain scission rate r

                    
                    bonds_register[pot_bond, 0] = 0  # Bond is broken!
                    bond_broken = True

                    ##                   print('Bond is broken')
                    ##                   sys.pause()
####                    if (bonds_register[pot_bond, 1] == 0):  # weak bond has broken
####                        weak_bond_broken = weak_bond_broken + 1
####                    elif (bonds_register[pot_bond, 1] == 1):  # strong bond has broken
####                        strong_bond_broken = strong_bond_broken + 1
                    strong_bond_broken = strong_bond_broken + 1

                        

                    t = t + t_KMC
######                    if (index % step == 0):
######                        f2.write('%5d  %5d  %5d  %0.4E  %0.4E  %0.4E  %0.4E  %0.4E  %5d  %5d  %5d\n' % (
######                            bonds_register[pot_bond, 2], bonds_register[pot_bond, 3],
######                            bonds_register[pot_bond, 4], bonds_register[pot_bond, 5], bonds_register[pot_bond, 6],
######                            t, t_KMC, vmax, len(active_bonds[0]), weak_bond_broken, strong_bond_broken))
######                        f2.flush()
                    # Local Relaxation -- If the bond-broken created a dangling end system
                    # then make the force on the remaining fourth bond

                    
                    link_1 = bonds_register[pot_bond, 3]
                    conn = get_link_bonds(link_1, bonds_register)

                    if (len(conn) == 3):
                        if (conn[list(conn)[0]] == 0 and conn[list(conn)[1]] == 0):
                            bonds_register[list(conn)[2], 6] = 0

                    elif (len(conn) == 2):
                        if (conn[list(conn)[0]] == 0):
                            bonds_register[list(conn)[1], 6] = 0

                    elif(len(conn)==1):
                        bonds_register[list(conn)[0], 6] = 0

                    link_2 = bonds_register[pot_bond, 4]
                    conn = get_link_bonds(link_2, bonds_register)
                    if (len(conn) == 3):
                        if (conn[list(conn)[0]] == 0 and conn[list(conn)[1]] == 0):
                            bonds_register[list(conn)[2], 6] = 0

                    elif (len(conn) == 2):
                        if (conn[list(conn)[0]] == 0):
                            bonds_register[list(conn)[1], 6] = 0

                    elif (len(conn) == 1):
                        bonds_register[list(conn)[0], 6] = 0

##                    temp1 = np.where(self.conn_arr[int(link_1) - 1] == link_2 - 1)[0]
##                    self.conn_arr[int(link_1) - 1, temp1] = -1
##                    temp2 = np.where(self.conn_arr[int(link_2) - 1] == link_1 - 1)[0]
##                    self.conn_arr[int(link_2) - 1, temp2] = -1
                    



                else:
                    t = t + t_KMC

                
                
                # active bond- means that it is not broken!!
                active_bonds = np.where(bonds_register[:, 0] == 1)  # all active bonds
        ##                active_bonds_weak = np.where(bonds_register[:,0]==1 and bonds_register[:,1]==0) # active bond and weak
        ##                active_bonds_strong = np.where(bonds_register[:,0]==1 and bonds_register[:,1]==1) # active bond and strong

            '''  # no need to update active_bonds because nothing is breaking 

######        if (index % step == 0): f2.close()

        n_bonds_final = len(active_bonds[0])
        if (n_bonds_final < n_bonds_init):
            bonds_final = np.zeros((n_bonds_final, 4), dtype=int)
            bonds_final[:, 0:4] = bonds_register[active_bonds[0], 1:5].astype(int)
            self.bonds = bonds_final

        ##print('time, init bonds, final bonds = %6.4E, %5d, %5d' % (t, n_bonds_init, n_bonds_final))
        ##print('---------------------------------------------------------------')

        ##        self.chains_conn_to_crosslink = (np.ones((len(self.atoms[:, 0]), 4)) * (-1)).astype('int')

        # reassess which chains are connected to which crosslinkers- only if chains break, not otherwise
        ##        for idx in range(0, len(self.bonds[:, 0])):
        ##            [lnk_1, lnk_2] = self.bonds[idx, 2:4]
        ##            a = np.where(self.chains_conn_to_crosslink[lnk_1 - 1, :] == -1)[0][0]  # first element where it is -1
        ##            self.chains_conn_to_crosslink[lnk_1 - 1, a] = idx
        ##            a = np.where(self.chains_conn_to_crosslink[lnk_2 - 1, :] == -1)[0][0]  # first element where it is -1
        ##            self.chains_conn_to_crosslink[lnk_2 - 1, a] = idx

        return t, n_bonds_init, n_bonds_final, weak_bond_broken, strong_bond_broken



    def KMCbondbreak_step(self, tau, delta_t, pflag, index, frac_weak):

        # Material parameters:
        # beta = 1.0 -- All material params, U0 and sigma, are in units of kT.
        # Main array: Bonds_register = [Activity index, type, index, link1, link2, dist, rate(ri)]
        # All are active at the start (active = 1, break = 0)

        def get_link_bonds(link, bonds_register):

            conn = {}
            a1 = np.where(bonds_register[:, 3] == link)
            a2 = np.where(bonds_register[:, 4] == link)
            a = np.concatenate((a1[0], a2[0]))
            a = np.unique(a)
            for i in range(0, len(a)):
                if (bonds_register[a[i], 0] == 1):
                    conn.update({a[i]: bonds_register[a[i], 5]})

            conn = dict(sorted(conn.items(), key=lambda x: x[1]))

            return conn

        ftype = self.ftype
        n_bonds = len(self.bonds[:, 0])
        bonds_register = np.zeros((n_bonds, 7))
        bonds_register[:, 0] = 1
        bonds_register[:, 1:5] = self.bonds
        dist = self.bondlengths(self.atoms, self.bonds, self.xlo,self.xhi,self.ylo,self.yhi,self.zlo,self.zhi)
        bonds_register[:, 5] = dist[:, 3]

        step = 10
        # File to write bond broken stats
######        if (index % step == 0):
######            directory = './' + str(int(100 * frac_weak)) + '/'
######            filename = 'bondbroken_%d.txt' % (index)
######            file_path = os.path.join(directory, filename)
######            if not os.path.isdir(directory):
######                os.mkdir(directory)
######            f2 = open(file_path, 'w')
######            f2.write(
######                '#type, atom1, atom2, length, rate(v), t, t_KMC, vmax, active bonds, num_weak_bond_broken, num_strong_bond_broken\n')

            # Write probability values in a file (at every KMC call)
        if (pflag == 1):
            prob_file = 'prob_%d.txt' % (index)
            directory = './' + str(int(100 * frac_weak)) + '/'
            filename = prob_file
            file_path = os.path.join(directory, filename)
            if not os.path.isdir(directory):
                os.mkdir(directory)
            fl1 = open(file_path, 'w')

        bonds = self.bonds
        ctype=1 ## setting this here because we are not using different bond types here
        parameters = self.parameters
        N = parameters[ctype, 0]  # bonds[i,0] gives the ctype
        b = parameters[ctype, 1]
        K = parameters[ctype, 2]
        fit_param = parameters[ctype, 3]
        E_b = parameters[ctype, 4]
        U0_kT = parameters[ctype, 5]
        Nb = N * b

            
        for i in range(0, n_bonds):
######            bonds = self.bonds
######            ctype = bonds[i, 0]
######            parameters = self.parameters
######            N = parameters[ctype, 0]  # bonds[i,0] gives the ctype
######            b = parameters[ctype, 1]
######            K = parameters[ctype, 2]
######            fit_param = parameters[ctype, 3]
######            E_b = parameters[ctype, 4]
######            U0_kT = parameters[ctype, 5]
######            Nb = N * b
            r = bonds_register[i, 5]
            if (r > 0):
                [fbond, lam_b] = self.get_bondforce(r, i)
            else:
                fbond = 0.0

            ##            fit_param = 1
            fbkT = -fbond * r / K
            bonds_register[i, 6] = math.exp(-U0_kT + fbkT * fit_param)
            if (pflag == 1): fl1.write('%i %i %i %i %i %6.4f %6.4f\n' % (bonds_register[i, 0],
                                                                         bonds_register[i, 1], bonds_register[i, 2],
                                                                         bonds_register[i, 3],
                                                                         bonds_register[i, 4], bonds_register[i, 5],
                                                                         bonds_register[i, 6]))

        if (pflag == 1): fl1.close()

        active_bonds = np.where(bonds_register[:, 0] == 1)
        n_bonds_init = len(active_bonds[0])
        vmax = max(bonds_register[active_bonds[0], 6])  # same as rmax in paper
        if (vmax == 0): vmax = 1e-12
        # if fbkT = 0, vmax = exp(-56). This number below the machine precison.
        # hence, we assign a small detectable number, vmax = 10^{-12}.
        # Essentially, it implies that bond breaking rate is very low, or
        # t = 1/(vmax*nbonds) is very high compare to del_t and hence it will not
        # enter the KMC bond breaking loop

        t = 1 / (vmax * len(active_bonds[0]))

        bond_broken = False

        weak_bond_broken = 0  # number of weak (and strong) bonds broken
        strong_bond_broken = 0
        ##print('KMC statistics:')
        ##print('Max rate, Active bonds, and t_KMC = %6.4E, %5d, %6.4E' % (vmax, len(active_bonds[0]), t))
        if (len(active_bonds[0])>0):#(t < delta_t):
            t = 0
            num_broken_this_KMC=0
            while (num_broken_this_KMC<1): #(t < delta_t):
                ##                print('inside KMC t')
                ##                sys.pause()
                t_KMC = 1 / (vmax * len(active_bonds[0]))
                vmax = max(bonds_register[active_bonds[0], 6])
                ##                random.seed(10)
                bond_index = random.randint(0, len(active_bonds[0]) - 1)
                pot_bond = active_bonds[0][bond_index]
                ##                random.seed(10)
                rnd_num = random.uniform(0, 1)
                if ((bonds_register[
                         pot_bond, 6] / vmax) > rnd_num):  # bonds_register[pot_bond,6] is chain scission rate r
                    bonds_register[pot_bond, 0] = 0  # Bond is broken!
                    num_broken_this_KMC=num_broken_this_KMC+1
                    ##bond_broken = True

                    ##                   print('Bond is broken')
                    ##                   sys.pause()
####                    if (bonds_register[pot_bond, 1] == 0):  # weak bond has broken
####                        weak_bond_broken = weak_bond_broken + 1
####                    elif (bonds_register[pot_bond, 1] == 1):  # strong bond has broken
####                        strong_bond_broken = strong_bond_broken + 1

                    strong_bond_broken = strong_bond_broken + 1

                    t = t + t_KMC
######                    if (index % step == 0):
######                        f2.write('%5d  %5d  %5d  %0.4E  %0.4E  %0.4E  %0.4E  %0.4E  %5d  %5d  %5d\n' % (
######                            bonds_register[pot_bond, 2], bonds_register[pot_bond, 3],
######                            bonds_register[pot_bond, 4], bonds_register[pot_bond, 5], bonds_register[pot_bond, 6],
######                            t, t_KMC, vmax, len(active_bonds[0]), weak_bond_broken, strong_bond_broken))
######                        f2.flush()
                    # Local Relaxation -- If the bond-broken created a dangling end system
                    # then make the force on the remaining fourth bond
                    link_1 = bonds_register[pot_bond, 3]
                    conn = get_link_bonds(link_1, bonds_register)

                    if (len(conn) == 3):
                        if (conn[list(conn)[0]] == 0 and conn[list(conn)[1]] == 0):
                            bonds_register[list(conn)[2], 6] = 0

                    elif (len(conn) == 2):
                        if (conn[list(conn)[0]] == 0):
                            bonds_register[list(conn)[1], 6] = 0

                    elif(len(conn)==1):
                        bonds_register[list(conn)[0], 6] = 0

                    link_2 = bonds_register[pot_bond, 4]
                    conn = get_link_bonds(link_2, bonds_register)
                    if (len(conn) == 3):
                        if (conn[list(conn)[0]] == 0 and conn[list(conn)[1]] == 0):
                            bonds_register[list(conn)[2], 6] = 0

                    elif (len(conn) == 2):
                        if (conn[list(conn)[0]] == 0):
                            bonds_register[list(conn)[1], 6] = 0

                    elif (len(conn) == 1):
                        bonds_register[list(conn)[0], 6] = 0

##                    temp1 = np.where(self.conn_arr[int(link_1) - 1] == link_2 - 1)[0]
##                    self.conn_arr[int(link_1) - 1, temp1] = -1
##                    temp2 = np.where(self.conn_arr[int(link_2) - 1] == link_1 - 1)[0]
##                    self.conn_arr[int(link_2) - 1, temp2] = -1


                else:
                    t = t + t_KMC
                # active bond- means that it is not broken!!
                active_bonds = np.where(bonds_register[:, 0] == 1)  # all active bonds
        ##                active_bonds_weak = np.where(bonds_register[:,0]==1 and bonds_register[:,1]==0) # active bond and weak
        ##                active_bonds_strong = np.where(bonds_register[:,0]==1 and bonds_register[:,1]==1) # active bond and strong

######        if (index % step == 0): f2.close()

        n_bonds_final = len(active_bonds[0])
        if (n_bonds_final < n_bonds_init):
            bonds_final = np.zeros((n_bonds_final, 4), dtype=int)
            bonds_final[:, 0:4] = bonds_register[active_bonds[0], 1:5].astype(int)
            self.bonds = bonds_final

        ##print('time, init bonds, final bonds = %6.4E, %5d, %5d' % (t, n_bonds_init, n_bonds_final))
        ##print('---------------------------------------------------------------')

        ##        self.chains_conn_to_crosslink = (np.ones((len(self.atoms[:, 0]), 4)) * (-1)).astype('int')

        # reassess which chains are connected to which crosslinkers- only if chains break, not otherwise
        ##        for idx in range(0, len(self.bonds[:, 0])):
        ##            [lnk_1, lnk_2] = self.bonds[idx, 2:4]
        ##            a = np.where(self.chains_conn_to_crosslink[lnk_1 - 1, :] == -1)[0][0]  # first element where it is -1
        ##            self.chains_conn_to_crosslink[lnk_1 - 1, a] = idx
        ##            a = np.where(self.chains_conn_to_crosslink[lnk_2 - 1, :] == -1)[0][0]  # first element where it is -1
        ##            self.chains_conn_to_crosslink[lnk_2 - 1, a] = idx

        return t, n_bonds_init, n_bonds_final, weak_bond_broken, strong_bond_broken




