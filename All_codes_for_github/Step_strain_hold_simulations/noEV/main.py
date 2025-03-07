#!/usr/local/bin/env python
# -*- coding: utf-8 -*-

"""
#######################################
#                                     #
#-- Fracture Simulation of Networks --#
### Mixed bond fracture update: Devosmita Sen###
########## Original: Akash Arora#######
#------  March 2023  --------#
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
        Assign bond dissociation energy of chains according to %S (percent of strong (or weak (frac_weak)) chains in the system
     
     2. Force relaxtion of network using Fast Inertial Relaxation Engine (FIRE) 
        to obtain the equilibrium positions of crosslinks (min-energy configuration)

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
#file_dir = os.path.dirname(__file__)
#sys.path.append(file_dir)
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append('../')
#print(sys.path)
import time
import math
import random
import matplotlib
import numpy as np
##import ioLAMMPS
##import netgen
##from relax import Optimizer
from numpy import linalg as LA
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import param as p
import shutil
'''
###########parameters##################
U0_low=1
U0_high=5

N_low=12.0
N_high=12.0

##frac_weak=0.0

L=30 # corresponding to loop frac~=0.05
lam_max=10
del_t=0.01
e_rate=5
n_chains=500

b_low=1.0
b_high=1.0

K_low=1.0
K_high=1.0

fit_param_low=1.0
fit_param_high=1.0

E_b_low=1200.0
E_b_high=1200.0

func=4

tol=0.01
max_itr = 100000
write_itr = 10000
wrt_step = 500
##################################################
'''
#random.seed(a=500)
random.seed(a=None, version=2)
##random.seed(10)
##random.seed(10)
print('First random number of this seed: %d'%(random.randint(0, 10000))) 
# This is just to check whether different jobs have different seeds
##global parameters
parameters=np.zeros([2,6]) # N, b, K, fit_param, E_b,U0
#parameters[0,:]=np.array([12,1.0,1.0,1.0,1200,56.7578])
#parameters[1,:]=np.array([12,1.0,1.0,1.0,1200,283.789])
##parameters[0,:]=np.array([p.N_low,p.b_low,p.K_low,p.fit_param_low,p.E_b_low,p.U0_low])
##parameters[1,:]=np.array([p.N_high,p.b_high,p.K_high,p.fit_param_high,p.E_b_high,p.U0_high])


parameters[0,:]=np.array([p.N_low,p.b_low,p.K_low,p.fit_param_low,p.E_b_low,p.U0_low])
parameters[1,:]=np.array([p.N_high,p.b_high,p.K_high,p.fit_param_high,p.E_b_high,p.U0_high])

##frac_weak=0.0
##frac_weak=p.frac_weak
#%#non deterministic step for assignning chain_type
##my_task_id = int(sys.argv[1])
##num_tasks = int(sys.argv[2])
frac_weak_array_py=[0.0]##,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
##frac_weak_array_py=frac_weak_array_py[my_task_id-1:len(frac_weak_array_py):num_tasks]      

for frac_weak in frac_weak_array_py:
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

   netgen_flag = 1
   swell = 0
   if(netgen_flag==0):

      vflag = 0
   ##   N = 12   
      print('--------------------------')   
      print('----Reading Network-------')   
      print('--------------------------')
      
      filename = "network.txt"
      file_path = os.path.join(directory, filename)
      if not os.path.isdir(directory):
         os.mkdir(directory)  
      [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
              atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS(file_path, vflag, frac_weak)
      
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
      netgen.generate_network(prob, func, parameters,L, l0, n_chains, n_links, frac_weak)
      directory = './'+str(int(100*frac_weak))+'/'
      filename = 'network.txt'
      file_path = os.path.join(directory, filename)
      if not os.path.isdir(directory):
         os.mkdir(directory)  
      
      [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
              atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS(file_path,0,frac_weak)

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
   filename = 'stress_continuous'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)  
   fstr=open(file_path,'w')
   fstr.write('#Lx, Ly, Lz, lambda, FE, deltaFE, st[0], st[1], st[2], st[3], st[4], st[5], t_KMC\n') 

   filename = 'strand_lengths_continuous'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)      
   flen=open(file_path,'w')
   flen.write('#lambda, ave(R), max(R)\n') 


   filename = 'KMC_stats_continuous'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)
   fkmc=open(file_path,'w')
   fkmc.write('#lambda, init bonds, final bonds, weak_bonds_broken,strong_bonds_broken\n') 
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

   mymin = Optimizer(atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, r0, parameters, 'Mao')
   [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log.txt')
   directory = './'+str(int(100*frac_weak))+'/'
   if(swell==1):
      
      filename = 'restart_network_01.txt'
      file_path = os.path.join(directory, filename)
      if not os.path.isdir(directory):
         os.mkdir(directory)  
   
      ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                     mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
      # Swelling the network to V = 2
      scale_x = 1.26
      scale_y = 1.26
      scale_z = 1.26
      mymin.change_box(scale_x, scale_y, scale_z)    
      [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log.txt')


   filename = 'restart_network_0.txt'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)  
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                     mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

   dist = mymin.bondlengths()
   Lx0 = mymin.xhi-mymin.xlo
   BE0 = e
   t_KMC=0
   [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
   fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f %7.4f\n' \
                             %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx, t_KMC))
   fstr.flush()

   flen.write('%7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
   flen.flush()

   fkmc.write('%7.4f  %5i  %5i %5i %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds, n_bonds,0,0))
   fkmc.flush()


   filename = 'restart_network_0.txt'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory) 
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                     mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
   
   #-------------------------------------#
   # Tensile deformation: lambda/scales  #
   #-------------------------------------#
##   sys.pause()
   steps = int((lam_max-1)/(erate*del_t))
   print('Deformation steps = ',steps)
   begin_break = -1         # -1 implies that bond breaking begins right from start
   #begin_break = n_steps   # implies bond breaking will begin after n_steps of deformation

   for i in range(0,steps):

       scale_x = (1+(i+1)*erate*del_t)/(1+i*erate*del_t)
       scale_y = scale_z = 1.0/math.sqrt(scale_x)
       mymin.change_box(scale_x, scale_y, scale_z)    
       [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log.txt')
       [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
       fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                                        %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                                     (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx, t_KMC)) 
       fstr.flush()

       dist = mymin.bondlengths()
       flen.write('%7.4fn'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
       flen.flush()
      
       if((i+1)%wrt_step==0): 
         filename = 'restart_network_%d.txt' %(i+1)
         file_path = os.path.join(directory, filename)
         if not os.path.isdir(directory):
            os.mkdir(directory) 
         ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,
                                              mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

       if(i > begin_break):
         # U0, tau, del_t, pflag, index
##         sys.pause()
         [t, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken] = mymin.KMCbondbreak( tau, del_t, 0, i+1,frac_weak)
         t_KMC=t_KMC+t
         if(n_bonds_final<n_bonds_init):
            print('bond broken in function')
##            sys.pause()
         
         fkmc.write('%7.4f  %5i  %5i  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken))
         fkmc.flush()
    
   #---------------------------------#
   #     Final Network Properties    #
   #---------------------------------#
   [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log'+str(100*frac_weak)+'.txt')
   [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
   fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                                    %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                                 (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx, t_KMC)) 
   fstr.flush()

   dist = mymin.bondlengths()
   flen.write('%7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
   flen.flush()

   fkmc.write('%7.4f  %5i  %5i %5i %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken))
   fkmc.flush()
   
   filename = 'restart_network_%d.txt' %(i+1)
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory) 
   
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,
                                          mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

   fstr.close()
   flen.close()
   fkmc.close()

   ## STEP SIMULATION STARTING NOW  #######

   directory = './'+str(int(100*frac_weak))+'/'
   filename = 'stress_step'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)  
   fstr=open(file_path,'w')
   fstr.write('#Lx, Ly, Lz, lambda, FE, deltaFE, st[0], st[1], st[2], st[3], st[4], st[5], t_KMC\n')


   filename = 'strand_lengths_step'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)      
   flen=open(file_path,'w')
   flen.write('#lambda, ave(R), max(R)\n') 


   filename = 'KMC_stats_step'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)
   fkmc=open(file_path,'w')
   fkmc.write('#lambda, init bonds, final bonds, weak_bonds_broken,strong_bonds_broken\n')


   #-------------------------------------#
   # Tensile deformation: lambda/scales  #
   #-------------------------------------#
##   sys.pause()
##   steps = int((lam_max-1)/(erate*del_t))

   itr_start=steps
   steps_max=len(mymin.bonds[:,0])
##   print('Deformation steps = ',steps)
   begin_break = -1         # -1 implies that bond breaking begins right from start
   #begin_break = n_steps   # implies bond breaking will begin after n_steps of deformation
   t_KMC=0
   t=t_KMC
   for i in range(itr_start+1,steps_max+itr_start+1):
       [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log'+str(100*frac_weak)+'.txt')
       [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
       fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                                    %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                                 (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx, t_KMC)) 
       fstr.flush()
   
       dist = mymin.bondlengths()
       flen.write('%7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
       flen.flush()


       if((i+1)%wrt_step==0 or i==1): 
         filename = 'restart_network_%d.txt' %(i+1)
         file_path = os.path.join(directory, filename)
         if not os.path.isdir(directory):
            os.mkdir(directory) 
         ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,
                                          mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

       if(i > begin_break):
         # U0, tau, del_t, pflag, index
##         sys.pause()
         [t, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken] = mymin.KMCbondbreak_step( tau, del_t, 0, i+1,frac_weak)
         t_KMC=t_KMC+t
         if(n_bonds_final<n_bonds_init):
            print('bond broken in function')
##            sys.pause()
         
         fkmc.write('%7.4f  %5i  %5i  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken))
         fkmc.flush()

   #---------------------------------#
   #     Final Network Properties    #
   #---------------------------------#

   [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log'+str(100*frac_weak)+'.txt')
   [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
   fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                                    %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                                 (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx, t_KMC)) 
   fstr.flush()

   dist = mymin.bondlengths()
   flen.write('%7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
   flen.flush()


   fkmc.write('%7.4f  %5i  %5i %5i %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken))
   fkmc.flush()
   
   filename = 'restart_network_%d.txt' %(i+1)
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory) 
   
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,
                                          mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

   fstr.close()
##   fstr2.close()
   flen.close()
   fkmc.close()
   sys.path.remove(file_dir)
