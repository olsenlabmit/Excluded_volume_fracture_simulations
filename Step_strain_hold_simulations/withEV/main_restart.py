#!/usr/local/bin/env python
# -*- coding: utf-8 -*-
"""
#######################################
#                                     #
#Coarse grained fracture Simulation of networks with Excluded Volume#
### Devosmita Sen ###
#------  August 2024  --------#
#                                     #
#######################################
System: A2+B4
A=chains
B=crosslinkers
 Overall Framework (Steps):
     1. Generate a Network following the algorithm published
        by AA Gusev, Macromolecules, 2019, 52, 9, 3244-3251
        if gen_net = 0, then it reads topology from user-supplied 
        network.txt file present in this folder

        Parameter epsilon denotes the strength of Excluded Volume potential
        This parameter has been optimized to ensure network does not shrink or swell upon equilibration
     
     2. Force relaxtion of network using Fast Inertial Relaxation Engine (FIRE) 
        to obtain the equilibrium positions of crosslinks (min-energy configuration)
        Total force consists of elastic attractive forces and repulsive excluded volume forces

     3. Compute Properties: Energy, Gamma (prestretch), and 
        Stress (all 6 componenets) 
     
     4. Deform the network (tensile) in desired direction by 
        strain format by supplying lambda_x, lambda_y, lambda_z

     5. Break bonds using Kintetic Theory of Fracture (force-activated KMC) 
        presently implemented algorithm is ispired by 
        Termonia et al., Macromolecules, 1985, 18, 2246

     6. Repeat steps 2-5 until the given extension (lam_total) is achived OR    
        stress decreases below a certain (user-specified) value 
        indicating that material is completey fractured.
"""

import os.path
import sys

import time
import math
import random
import matplotlib
import numpy as np

from numpy import linalg as LA
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

import param as p
import shutil
import pickle

import networkx as nx
start_time=time.time()
#random.seed(a=500)
random.seed(a=None, version=2)
##random.seed(10)
##random.seed(10)
print('First random number of this seed: %d'%(random.randint(0, 10000))) 
# This is just to check whether different jobs have different seeds
##global parameters
parameters=np.zeros([2,6]) # N, b, K, fit_param, E_b,U0
parameters[0,:]=np.array([p.N_low,p.b_low,p.K_low,p.fit_param_low,p.E_b_low,p.U0_low])
parameters[1,:]=np.array([p.N_high,p.b_high,p.K_high,p.fit_param_high,p.E_b_high,p.U0_high])

frac_weak_array_py=[0.0]#,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

def bondlength_no_pbc(links, link_1, link_2):
 
    
      lk = links[link_1,:] - links[link_2,:]
      
##          lk[0] = lk[0] - int(round(lk[0]/Lx))*Lx
##          lk[1] = lk[1] - int(round(lk[1]/Ly))*Ly
##          lk[2] = lk[2] - int(round(lk[2]/Lz))*Lz
            
      dist= LA.norm(lk)

      return dist

def bondlength(links, link_1, link_2):
      Lx=mymin.xhi-mymin.xlo
      Ly=mymin.yhi-mymin.ylo
      Lz=mymin.zhi-mymin.zlo
      
    
      lk = links[link_1,:] - links[link_2,:]
      
      lk[0] = lk[0] - int(round(lk[0]/Lx))*Lx
##      lk[1] = lk[1] - int(round(lk[1]/Ly))*Ly
##      lk[2] = lk[2] - int(round(lk[2]/Lz))*Lz
            
      dist= LA.norm(lk)

      return dist


def readLAMMPS_restart(filename, vflag,frac_weak):

   f1=open(filename,"r")

   line1 = f1.readline()
   line2 = f1.readline()

   line3 = f1.readline()
   line3 = line3.strip()
   n_links = int(line3.split(" ")[0])
 
   line4 = f1.readline()
   line4 = line4.strip()
   atom_types = int(line4.split(" ")[0])

   line5 = f1.readline()
   line5 = line5.strip()
   n_chains = int(line5.split(" ")[0])

   line6 = f1.readline()\
           
   line6 = line6.strip()
   bond_types = int(line6.split(" ")[0])

   links_unsort  = np.zeros((n_links,4))
   links   = np.zeros((n_links,3), dtype = float)
   chains  = np.full((n_chains,4), -1, dtype = int)
   mass    = np.zeros(atom_types, dtype = float)

   line7 = f1.readline()
   line8 = f1.readline()
   line8 = line8.strip()
   xlo = float(line8.split(" ")[0])
   xhi = float(line8.split(" ")[1])

   line9 = f1.readline()
   line9 = line9.strip()
   ylo = float(line9.split(" ")[0])
   yhi = float(line9.split(" ")[1])

   line10 = f1.readline()
   line10 = line10.strip()
   zlo = float(line10.split(" ")[0])
   zhi = float(line10.split(" ")[1])


   for i in range (0, 3):
       f1.readline()
   
   for i in range(0, atom_types):
       line = f1.readline()
       line = line.strip()
       mass[i] = float(line.split(" ")[1])

   f1.close()


   links_unsort = np.genfromtxt(filename, usecols=(0,3,4,5), skip_header=18, max_rows=n_links)

   for i in range(0, n_links):
       index = int(links_unsort[i,0])
       links[index-1,:] = links_unsort[i,1:4]

   if(vflag==0):
      data= np.genfromtxt(filename,usecols=(1,2,3), skip_header=17+n_links+3, max_rows=n_chains)
      chains[:,0]=data[:,0]-np.ones(len(chains)) # ctype
      chains[:,1]=np.ones(len(chains)) # column of ones
      chains[:,2:4]=data[:,1:3] # cl1,cl2
   elif(vflag==1):
      data= np.genfromtxt(filename,usecols=(1,2,3), skip_header=17+2*n_links+2*3, max_rows=n_chains)
      chains[:,0]=data[:,0]
      chains[:,1]=np.ones(len(chains))
      chains[:,2]=data[:,1]
      chains[:,3]=data[:,2]
   else:
      print("Invalid Velocity Flag")


   
##   print(chains)
   directory = './'+str(int(100*frac_weak))+'/'
   filename = 'primary_loops'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)  
   loop_atoms = np.genfromtxt(file_path, usecols=(1), skip_header=0)
   loop_atoms.tolist()
####   G=nx.MultiGraph()
####   for chain in chains[:,2:4]:
####      G.add_edge(chain[0],chain[1])
######      G.add_edge()

   return xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, links, chains, atom_types, bond_types, mass, loop_atoms




#-------------------------------------#
#       Simulation Parameters         #
#-------------------------------------#

#N  = 12
##chain_type=
##Nb = N
##K  = 1.0
r0 = 0.0
#U0  = 1 # not used anywhere inside the functions, just passed to function as parameters
tau = 1# not used anywhere inside the functions, just passed to function as parameters
#del_t=0.008
del_t=p.del_t
erate = p.e_rate
#lam_max = 25
lam_max = p.lam_max
lam_step=p.lam_step
##tol = 0.01
##max_itr = 100000
##write_itr = 10000
##wrt_step = 500
tol = p.tol
max_itr = p.max_itr
write_itr = p.write_itr
wrt_step = p.wrt_step
steps_max = p.n_chains ##int((lam_max-1)/(erate*del_t))

import re

def find_max_restart_number(folder_path):
    """
    Searches for files of type 'restart_network_*.txt' in the specified folder
    and returns the maximum number replacing '*'.

    Parameters:
        folder_path (str): Path to the folder to search.

    Returns:
        int: The maximum number found, or None if no such files are present.
    """
    max_number = None
    pattern = re.compile(r"restart_network_(\d+)\.txt")

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if max_number is None or number > max_number:
                max_number = number

    return max_number


for frac_weak in frac_weak_array_py:
   ## find restart_ite
   directory = './'+str(int(100*frac_weak))
   restart_ite = find_max_restart_number (directory)
   print('restart_ite=',restart_ite)
   hold_step=False
   
   if(restart_ite>int((lam_step-1)/(erate*del_t))):
       hold_step=True
   

   ##stop
   '''
   directory = './'+str(int(100*frac_weak))+'/'
   file_path = os.path.join(directory, filename)
      if not os.path.isdir(directory):
         os.mkdir(directory) 
   '''
   directory = './function_files/'
   orig_dir = os.path.dirname(directory)
   files=os.listdir(orig_dir)
   directory = './'+str(int(100*frac_weak))+'/'
   if not os.path.isdir(directory):
      os.mkdir(directory)

   for fname in files:
     
    # copying the files to the
    # destination directory
       shutil.copy2(os.path.join(orig_dir,fname), directory)

# now add path to frac_weak directory
   file_dir = os.path.dirname(directory)
   sys.path.append(file_dir)
   import ioLAMMPS
   import netgen
   from relax import Optimizer

   
   # in this, which parameter is assigned is given by variable chain_type
   G=nx.Graph()
   netgen_flag = 0
   swell = 0
   if(netgen_flag==0):
      

      vflag = 0
   ##   N = 12   
      print('--------------------------')   
      print('----Reading Network-------')   
      print('--------------------------')
      
      filename = "restart_network_"+str(restart_ite)+".txt"
      file_path = os.path.join(directory, filename)
      if not os.path.isdir(directory):
         os.mkdir(directory)  
      [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
              atom_types, bond_types, mass, loop_atoms, G] = ioLAMMPS.readLAMMPS_restart(file_path, vflag, frac_weak,G)
      print(xlo,xhi)

      

      directory = './'+str(int(100*frac_weak))+'/'
      n_chains=n_bonds
      n_links=n_atoms
      L=xhi-xlo
      ##filename = 'conn_arr.pckl'
      #file_path = os.path.join(directory, filename)
      ##conn_arr=pickle.load(file_path)
      print('xlo, xhi',xlo, xhi) 
      print('ylo, yhi',ylo, yhi) 
      print('zlo, zhi',zlo, zhi) 
      print('n_atoms', n_atoms) 
      print('n_bonds', n_bonds) 
      print('atom_types = ', atom_types) 
      print('bond_types = ', bond_types) 
      print('mass = ', mass) 
      print('primary loops = ', len(loop_atoms)) 
      print('--------------------------')   

   elif(netgen_flag==1):

   ##   func = 4
      func=p.func
   ##   N    = 12
   ##   rho  = 3
      l0   = 1
      prob = 1.0
      #n_chains=10000
      n_chains  = p.n_chains
      n_links   = int(2*n_chains/func)
      #L = 28
      L=p.L
      print(prob, func, parameters,L, l0, n_chains, n_links)
      conn_arr=netgen.generate_network(prob, func, parameters,L, l0, n_chains, n_links, frac_weak)
      directory = './'+str(int(100*frac_weak))+'/'
      filename = 'network.txt'
      file_path = os.path.join(directory, filename)
      if not os.path.isdir(directory):
         os.mkdir(directory)  
      
      [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
              atom_types, bond_types, mass, loop_atoms,G] = ioLAMMPS.readLAMMPS(file_path,0,frac_weak,G)

      
   ##   print('atoms \n',atoms)
   ##   print('bonds \n',bonds)
   ##   stop
   else:
      print('Invalid network generation flag')

   ##stop
            
##   save_path =str(100*int(frac_weak))
##   completeName = os.path.join(save_path,"stress")         
   c=float(n_chains)/(L**3.0)
   b=1
   N=parameters[0,0]
   dim_conc=c*(b**3)*(N**1.5)
   print('dim_conc',dim_conc)
   print('Loop fraction wrt chains=',(len(loop_atoms)/n_chains)*100,'%')
   #stop
   directory = './'+str(int(100*frac_weak))+'/'
   filename = 'stress'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)  
   fstr=open(file_path,'a')
   ##fstr.write('#Lx, Ly, Lz, lambda, FE, deltaFE, st[0], st[1], st[2], st[3], st[4], st[5], t_KMC\n')

   filename = 'stress_step'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)  
   fstr2=open(file_path,'a')
   ##fstr2.write('#Lx, Ly, Lz, lambda, FE, deltaFE, st[0], st[1], st[2], st[3], st[4], st[5], t_KMC\n')

####   filename = 'stress_elastic'
####   file_path = os.path.join(directory, filename)

####   fstr2=open(file_path,'w')
####   fstr2.write('#Lx, Ly, Lz, lambda, FE, deltaFE, st[0], st[1], st[2], st[3], st[4], st[5]\n') 

####   filename = 'strand_lengths'
####   file_path = os.path.join(directory, filename)
####   if not os.path.isdir(directory):
####      os.mkdir(directory)      
####   flen=open(file_path,'w')
####   flen.write('#lambda, ave(R), max(R)\n') 


####   filename = 'KMC_stats'
####   file_path = os.path.join(directory, filename)
####   if not os.path.isdir(directory):
####      os.mkdir(directory)
####   fkmc=open(file_path,'w')
####   fkmc.write('#lambda, init bonds, final bonds, weak_bonds_broken,strong_bonds_broken\n') 
   #-------------------------------------#
   #       Simulation Parameters         #
   #-------------------------------------#

   #N  = 12
   ##chain_type=
   ##Nb = N
   ##K  = 1.0
   r0 = 0.0
   #U0  = 1 # not used anywhere inside the functions, just passed to function as parameters
   tau = 1# not used anywhere inside the functions, just passed to function as parameters
   #del_t=0.008
   del_t=p.del_t
   erate = p.e_rate
   #lam_max = 25
   lam_max = p.lam_max
   lam_step=p.lam_step
   ##tol = 0.01
   ##max_itr = 100000
   ##write_itr = 10000
   ##wrt_step = 500
   tol = p.tol
   max_itr = p.max_itr
   write_itr = p.write_itr
   wrt_step = p.wrt_step

   #-------------------------------------#
   #       First Force Relaxation        #
   #-------------------------------------#
##   stop

   #nx_num=500## for binning to scale the EV force

   
   
   mymin = Optimizer(atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, r0, parameters,p.epsilon, 'Mao')
   Lx=mymin.xhi-mymin.xlo
   Ly=mymin.yhi-mymin.ylo
   Lz=mymin.zhi-mymin.zlo
   mymin.yhi=mymin.yhi+Ly
   mymin.ylo=mymin.ylo-Ly
   mymin.zhi=mymin.zhi+Lz
   mymin.zlo=mymin.zlo-Lz

   Ly=mymin.yhi-mymin.ylo
   Lz=mymin.zhi-mymin.zlo

   print(mymin.xhi, mymin.xlo)
####   stop

   Lx0=p.L##Lx

   delx=1 ##Lx/nx_num ## for scaling the EV force

   nx_num=math.ceil(Lx0/delx)
   # readjust delx based on the nx_num value
   # this makes sure that the delx considered in every iteration remains ~constant, and there is also no errors due to truncating
   delx=Lx/nx_num
   # nx_num will be different in different iterations, and increasing as Lx increases
   # but the way the functions are written, this does not ass any extra computational complexity
   print('nx_num',nx_num)
   
   mymin.delx=delx

   ##wy_arr=np.ones(nx_num)*Lx0 ## initial value of width along y and z is juts the initial width, before any force relax
   ##wz_arr=np.ones(nx_num)*Lx0

####   wyz_arr_avg2=np.square(wy_arr+wz_arr)
   

   mymin.Lx0=Lx0
   mymin.Ly0=Lx0
   mymin.Lz0=Lx0
   ##Lyz0_avg2=(mymin.Ly0+mymin.Lz0)**2 ## this is the initial value, =Lx0
   
   ##factor=np.square(wy_arr+wz_arr)/Lyz0_avg2

   Gcc = sorted(nx.connected_components(G), key=len, reverse=True)

        # Print the nodes in the largest connected component
   #print(Gcc[0])
   largest_cc=np.array(list(Gcc[0]))-1##np.unique(mymin.bonds[:, 2:4])-1  ##np.array(list(Gcc[0]))-1
   largest_cc_EV=np.unique(mymin.bonds[:, 2:4])-1

   
   

   
##   conn_array=np.ones((n_atoms, func))*(-1)

   
   ##[e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr,True,True, factor,largest_cc_EV,'log.txt')

####   stop
   directory = './'+str(int(100*frac_weak))+'/'
  

   '''
   filename = 'restart_network_0_before_PBC_break.txt'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)  
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                     mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
   '''
   
   ##dist = mymin.bondlengths(mymin.atoms, mymin.bonds, mymin.xlo,mymin.xhi,mymin.ylo,mymin.yhi,mymin.zlo,mymin.zhi)
   Lx0 = p.L##mymin.xhi-mymin.xlo
   ##BE0 = e
   ##t_KMC=0


   ###np.savetxt(file_path,np.array([inv_volume]))
   if(hold_step==False):
      stress_data=np.genfromtxt('./0/stress',skip_header=True)
   else:
      stress_data=np.genfromtxt('./0/stress_step',skip_header=True)
   t_KMC=stress_data[-1,12] ## the last data point recorded  # this should be -2 because steps+1 data point is not there in the stesss data, hence it gets shifted by 1
   t=t_KMC
   print('t_KMC at restart',t_KMC)

   ##[pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure(inv_volume)
   ##[pxx_EV, pyy_EV, pzz_EV, pxy_EV, pyz_EV, pzx_EV] = mymin.compute_pressure_EV(largest_cc_EV, mymin.atoms,len(mymin.atoms[:,0]),mymin.rc,mymin.del_x,mymin.force_factor, mymin.epsilon, Lx,Ly,Lz, factor,Lx0,delx, mymin.pre_factor,N,inv_volume)##(mymin.bonds, mymin.atoms,len(mymin.atoms[:,0]),mymin.rc,mymin.del_x,mymin.force_factor, mymin.epsilon, Lx,Ly,Lz, inv_volume)
   ##fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                             ##%(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                              ##(mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx+pxx_EV, pyy+pyy_EV, pzz+pzz_EV, pxy+pxy_EV, pyz+pyz_EV, pzx+pzx_EV, t_KMC))
####   fstr2.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
####                             %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
####                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
####   fstr.flush()
####   fstr2.flush()

####   flen.write('%7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
####   flen.flush()

####   fkmc.write('%7.4f  %5i  %5i %5i %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds, n_bonds,0,0))
####   fkmc.flush()


   # increase the box dimensions in y and z so that there is no wrapping back
   '''
   filename = 'restart_network_0.txt'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)  
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                     mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)


   filename = 'restart_network_0.txt'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.xlo, mymin.xhi,  mymin.xlo, mymin.xhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
   '''
   
   xlo_init=mymin.xlo
   xhi_init=mymin.xhi


   ##dist = mymin.bondlengths(mymin.atoms, mymin.bonds, mymin.xlo,mymin.xhi,mymin.ylo,mymin.yhi,mymin.zlo,mymin.zhi)
   Lx0 = p.L##mymin.xhi-mymin.xlo
   
   ##BE0 = e
   ##[pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure(inv_volume)
   ##[pxx_EV, pyy_EV, pzz_EV, pxy_EV, pyz_EV, pzx_EV] = mymin.compute_pressure_EV(largest_cc_EV, mymin.atoms,len(mymin.atoms[:,0]),mymin.rc,mymin.del_x,mymin.force_factor, mymin.epsilon, Lx,Ly,Lz, factor,Lx0,delx, mymin.pre_factor,N,inv_volume)##(mymin.bonds, mymin.atoms,len(mymin.atoms[:,0]),mymin.rc,mymin.del_x,mymin.force_factor, mymin.epsilon, Lx,Ly,Lz, inv_volume)
   ##fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                             ##%(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                              ##(mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx+pxx_EV, pyy+pyy_EV, pzz+pzz_EV, pxy+pxy_EV, pyz+pyz_EV, pzx+pzx_EV, t_KMC))
   ##fstr.flush()

####   fstr2.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
####                             %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
####                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
####   fstr2.flush()

####   flen.write('%7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
####   flen.flush()

####   fkmc.write('%7.4f  %5i  %5i %5i %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds, n_bonds,0,0))
####   fkmc.flush()
   


  
   ylo_old=ylo
   yhi_old=yhi
   zlo_old=zlo
   zhi_old=zhi
   
   ##Lx=Lx0
   inv_volume=np.genfromtxt('./0/inv_vol.txt')##1/(Lx0**3)


   delx=Lx/nx_num ## for scaling the EV force
   mymin.delx=delx
##   print('G',G)
##   print(nx.connected_components(G))
   
##   stop
####   [y_min_arr,y_max_arr,z_min_arr,z_max_arr,xidx_arr]=mymin.get_width_test(nx_num, mymin.bonds, mymin.atoms, mymin.delx,largest_cc) # nx is the number of bins along x axis where we want to calculate the width

   ##[wy_arr,wz_arr]=mymin.get_width_init(nx_num, mymin.bonds, mymin.atoms, delx, largest_cc)
   ##Lyz0_avg2=np.square(np.mean(wy_arr)+np.mean(wz_arr))  # this is the initial width (after force relaxation step), and this remains constant throughout
   
   Lyz0_avg2=np.genfromtxt('./0/Lyz0_avg2.txt')
   ##factor=mymin.get_width(nx_num, mymin.bonds, mymin.atoms, mymin.delx,largest_cc,Lyz0_avg2) # nx is the number of bins along x axis where we want to calculate the width

   ##(nx_num, bonds, atoms, delx, largest_cc,Lyz0_avg2)
####   stop

   
             
####   mymin.Ly0=np.mean(wy_arr) ## average max width along y and z direction
####   mymin.Lz0=np.mean(wz_arr) ## Doing this such that the Ly0 is scaled by the actual width at the initial time step (since the width is based on the max -min estimate, it can be slightly higher than the actual width)
   ## here, I am normalizing wrt this effective width such that the EV forces are consistently normalized
####   stop
    
####   stop

             
   #-------------------------------------#
   # Tensile deformation: Continuous: lambda/scales  #
   #-------------------------------------#
##   sys.pause()
####   steps = int((lam_max-1)/(erate*del_t))

   steps = int((lam_step-1)/(erate*del_t))
   print('Deformation steps = ',steps)
   begin_break = -1         # -1 implies that bond breaking begins right from start
   #begin_break = n_steps   # implies bond breaking will begin after n_steps of deformation

   for i in range(restart_ite,steps):

       scale_x = (1+(i+1)*erate*del_t)/(1+i*erate*del_t)
       scale_y = scale_z = 1.0 ##1.0/math.sqrt(scale_x) # don't change the dimensions along y and z
       mymin.change_box(scale_x, scale_y, scale_z)
       scale_model=(1.0/math.sqrt(scale_x)) # model Ly according to poisson ratio of 0.5

##       xmid = (xlo + xhi) / 2
       ymid = (ylo_old + yhi_old) / 2
       zmid = (zlo_old + zhi_old) / 2

##       new_xlo = xmid + scale_x * (xlo - xmid)
       new_ylo = ymid + scale_model * (ylo_old - ymid)
       new_zlo = zmid + scale_model * (zlo_old - zmid)

##       new_xhi = xmid + scale_x * (xhi - xmid)
       new_yhi = ymid + scale_model * (yhi_old - ymid)
       new_zhi = zmid + scale_model * (zhi_old - zmid)

##       newLx = new_xhi - new_xlo
       newLy = new_yhi - new_ylo
       newLz = new_zhi - new_zlo


       ylo_old=new_ylo
       yhi_old=new_yhi
       zlo_old=new_zlo
       zhi_old=new_zhi
       Lx=mymin.xhi-mymin.xlo
       

       
       ##delx=Lx/nx ## for scaling the EV force
       ##mymin.delx=delx
       ##nx_num=math.ceil(Lx/delx)
       # readjust delx based on the nx_num value
       # this makes sure that the delx considered in every iteration remains ~constant, and there is also no errors due to truncating
       delx=Lx/nx_num
       ##print('nx_num', nx_num)
       ##print('delx',delx)
       mymin.delx=delx ## readjusted delx
       factor=mymin.get_width(nx_num, mymin.bonds, mymin.atoms, mymin.delx, largest_cc,Lyz0_avg2) # nx is the number of bins along x axis where we want to calculate the width
       ##(nx_num, bonds, atoms, delx, largest_cc,Lyz0_avg2)

       
             
       ## largest_cc remains the same throughout the simulation until complete fracture

             
       [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr,True,False, factor,largest_cc_EV,'log.txt')
       
      
       if((i+1)%wrt_step==0):
         ##stop
         BE0=e

         [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure(inv_volume)
         [pxx_EV, pyy_EV, pzz_EV, pxy_EV, pyz_EV, pzx_EV] = mymin.compute_pressure_EV(largest_cc_EV, mymin.atoms,len(mymin.atoms[:,0]),mymin.rc,mymin.del_x,mymin.force_factor, mymin.epsilon, Lx,Ly,Lz, factor,Lx0,delx, mymin.pre_factor,N,inv_volume)
         ##(mymin.bonds, mymin.atoms,len(mymin.atoms[:,0]),mymin.rc,mymin.del_x,mymin.force_factor, mymin.epsilon, Lx,Ly,Lz, inv_volume)
         
       
####         fstr2.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
####                             %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
####                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
####         fstr2.flush()

         ##dist = mymin.bondlengths(mymin.atoms, mymin.bonds, mymin.xlo,mymin.xhi,mymin.ylo,mymin.yhi,mymin.zlo,mymin.zhi)
####         flen.write('%7.4fn'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
####         flen.flush()

       
         filename = 'restart_network_%d.txt' %(i+1)
         file_path = os.path.join(directory, filename)
         if not os.path.isdir(directory):
            os.mkdir(directory) 
         ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, new_ylo, new_yhi, new_zlo, new_zhi,
                                              mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

       if(i > begin_break):
         # U0, tau, del_t, pflag, index
##         sys.pause()
         [t, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken] = mymin.KMCbondbreak_nobreak( tau, del_t, 0, i+1,frac_weak)
         t_KMC=t_KMC+t
         

       fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                             %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx+pxx_EV, pyy+pyy_EV, pzz+pzz_EV, pxy+pxy_EV, pyz+pyz_EV, pzx+pzx_EV, t_KMC))
       fstr.flush()
####         if(n_bonds_final<n_bonds_init):
####            print('bond broken in function')
##            sys.pause()
         
####         fkmc.write('%7.4f  %5i  %5i  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken))
####         fkmc.flush()

   #-------------------------------------#
   # Tensile deformation: Step hold: lambda/scales  #
   #-------------------------------------#
##   sys.pause()
####   steps = int((lam_max-1)/(erate*del_t))

####   steps = int((lam_max-lam_step)/(erate*del_t))


   itr_start=max(steps,restart_ite) ## if continuous was not completed, then restrt ite is lower and hence steps will be chosen, else restart_ite will be chosen
   steps_max=len(mymin.bonds[:,0])
####   print('Deformation steps = ',steps)
   begin_break = -1         # -1 implies that bond breaking begins right from start
   #begin_break = n_steps   # implies bond breaking will begin after n_steps of deformation

   scale_x = (1+((steps-1)+1)*erate*del_t)/(1+(steps-1)*erate*del_t)
   scale_y = scale_z = 1.0 ##1.0/math.sqrt(scale_x) # don't change the dimensions along y and z
   mymin.change_box(scale_x, scale_y, scale_z)
   scale_model=(1.0/math.sqrt(scale_x)) # model Ly according to poisson ratio of 0.5

##       xmid = (xlo + xhi) / 2
   ymid = (ylo_old + yhi_old) / 2
   zmid = (zlo_old + zhi_old) / 2

##       new_xlo = xmid + scale_x * (xlo - xmid)
   new_ylo = ymid + scale_model * (ylo_old - ymid)
   new_zlo = zmid + scale_model * (zlo_old - zmid)

##       new_xhi = xmid + scale_x * (xhi - xmid)
   new_yhi = ymid + scale_model * (yhi_old - ymid)
   new_zhi = zmid + scale_model * (zhi_old - zmid)

   newLy = new_yhi - new_ylo
   newLz = new_zhi - new_zlo

   for i in range(itr_start,steps_max+itr_start+1):
       '''
       scale_x = (1+(i+1)*erate*del_t)/(1+i*erate*del_t)
       scale_y = scale_z = 1.0 ##1.0/math.sqrt(scale_x) # don't change the dimensions along y and z
       mymin.change_box(scale_x, scale_y, scale_z)
       scale_model=(1.0/math.sqrt(scale_x)) # model Ly according to poisson ratio of 0.5

##       xmid = (xlo + xhi) / 2
       ymid = (ylo_old + yhi_old) / 2
       zmid = (zlo_old + zhi_old) / 2

##       new_xlo = xmid + scale_x * (xlo - xmid)
       new_ylo = ymid + scale_model * (ylo_old - ymid)
       new_zlo = zmid + scale_model * (zlo_old - zmid)

##       new_xhi = xmid + scale_x * (xhi - xmid)
       new_yhi = ymid + scale_model * (yhi_old - ymid)
       new_zhi = zmid + scale_model * (zhi_old - zmid)

##       newLx = new_xhi - new_xlo
       newLy = new_yhi - new_ylo
       newLz = new_zhi - new_zlo


       ylo_old=new_ylo
       yhi_old=new_yhi
       zlo_old=new_zlo
       zhi_old=new_zhi
       Lx=mymin.xhi-mymin.xlo

       
       ##delx=Lx/nx ## for scaling the EV force
       ##mymin.delx=delx
       ##nx_num=math.ceil(Lx/delx)
       # readjust delx based on the nx_num value
       # this makes sure that the delx considered in every iteration remains ~constant, and there is also no errors due to truncating
       delx=Lx/nx_num
       print('nx_num', nx_num)
       print('delx',delx)
       mymin.delx=delx ## readjusted delx
       '''
       factor=mymin.get_width(nx_num, mymin.bonds, mymin.atoms, mymin.delx, largest_cc,Lyz0_avg2) # nx is the number of bins along x axis where we want to calculate the width
       ##(nx_num, bonds, atoms, delx, largest_cc,Lyz0_avg2)
       
       
       '''
       for xidx in range(0,nx_num):
         if(wy_arr[xidx]==0 or wy_arr[xidx]==-1000):
             wy_arr[xidx]=wy_arr[xidx-1] ## wy_arr[xidx-1] must have been non zero, otherwise it would have been cauht at an earlier iteration
         if(wz_arr[xidx]==0 or wz_arr[xidx]==-1000):
             wz_arr[xidx]=wz_arr[xidx-1]
       '''      
       ## largest_cc remains the same throughout the simulation until complete fracture

             
       [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr,True,False, factor,largest_cc_EV,'log.txt')
       
      
       if((i+1)%wrt_step==0):

         [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure(inv_volume)
         [pxx_EV, pyy_EV, pzz_EV, pxy_EV, pyz_EV, pzx_EV] = mymin.compute_pressure_EV(largest_cc_EV, mymin.atoms,len(mymin.atoms[:,0]),mymin.rc,mymin.del_x,mymin.force_factor, mymin.epsilon, Lx,Ly,Lz, factor,Lx0,delx, mymin.pre_factor,N,inv_volume)
         ##(mymin.bonds, mymin.atoms,len(mymin.atoms[:,0]),mymin.rc,mymin.del_x,mymin.force_factor, mymin.epsilon, Lx,Ly,Lz, inv_volume)
         
       
####         fstr2.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
####                             %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
####                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
####         fstr2.flush()

         ##dist = mymin.bondlengths(mymin.atoms, mymin.bonds, mymin.xlo,mymin.xhi,mymin.ylo,mymin.yhi,mymin.zlo,mymin.zhi)
####         flen.write('%7.4fn'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
####         flen.flush()

       
         filename = 'restart_network_%d.txt' %(i+1)
         file_path = os.path.join(directory, filename)
         if not os.path.isdir(directory):
            os.mkdir(directory) 
         ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, new_ylo, new_yhi, new_zlo, new_zhi,
                                              mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

       if(i > begin_break):
         # U0, tau, del_t, pflag, index
##         sys.pause()
         [t, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken] = mymin.KMCbondbreak_step( tau, del_t, 0, i+1,frac_weak)
         t_KMC=t_KMC+t
       BE0=e
       fstr2.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                             %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx+pxx_EV, pyy+pyy_EV, pzz+pzz_EV, pxy+pxy_EV, pyz+pyz_EV, pzx+pzx_EV, t_KMC))
       fstr2.flush()
####         if(n_bonds_final<n_bonds_init):
####            print('bond broken in function')
##            sys.pause()
         
####         fkmc.write('%7.4f  %5i  %5i  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken))
####         fkmc.flush()

         
    
   #---------------------------------#
   #     Final Network Properties    #
   #---------------------------------#
   [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr,True,False, factor,largest_cc_EV,'log.txt')## mymin.fire_iterate(tol, max_itr, write_itr, True,False,factor, 'log'+str(100*frac_weak)+'.txt')
   [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure(inv_volume)
   [pxx_EV, pyy_EV, pzz_EV, pxy_EV, pyz_EV, pzx_EV] = mymin.compute_pressure_EV(largest_cc_EV, mymin.atoms,len(mymin.atoms[:,0]),mymin.rc,mymin.del_x,mymin.force_factor, mymin.epsilon, Lx,Ly,Lz, factor,Lx0,delx, mymin.pre_factor,N,inv_volume)#(mymin.bonds, mymin.atoms,len(mymin.atoms[:,0]),mymin.rc,mymin.del_x,mymin.force_factor, mymin.epsilon, Lx,Ly,Lz, inv_volume)
   fstr2.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                             %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx+pxx_EV, pyy+pyy_EV, pzz+pzz_EV, pxy+pxy_EV, pyz+pyz_EV, pzx+pzx_EV, t_KMC))

   fstr.flush()
   fstr2.flush()
 
####   fstr2.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
####                       %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
####                        (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
####   fstr2.flush()


   ##dist = mymin.bondlengths(mymin.atoms, mymin.bonds, mymin.xlo,mymin.xhi,mymin.ylo,mymin.yhi,mymin.zlo,mymin.zhi)
####   flen.write('%7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
####   flen.flush()

####   fkmc.write('%7.4f  %5i  %5i %5i %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken))
####   fkmc.flush()
   
   filename = 'restart_network_%d.txt' %(i+1)
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory) 
   # for this case, save the last time step as the model box size, so that that remains same for the following step simulation
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, new_ylo, new_yhi, new_zlo, new_zhi,
                                          mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
                                          # in the final file, save the sim box as the initial sim box- to ccompare the volume with the earlier result

   fstr.close()
####   flen.close()
####   fkmc.close()
####   fstr2.close()
####   sys.path.remove(file_dir)

end_time=time.time()
# Calculate elapsed time
elapsed_time = end_time - start_time

# Write only the elapsed time to a file
with open("timing_results_restart.txt", "w") as file:
    file.write(f"{elapsed_time:.6f}\n")
