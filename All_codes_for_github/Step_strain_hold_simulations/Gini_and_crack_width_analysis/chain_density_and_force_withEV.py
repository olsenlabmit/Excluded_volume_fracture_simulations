## plot the chain density distribution along the tensile axis and average force in all chains passing through bins along the tensile axis

import numpy as np
##import ioLAMMPS
##import ioLAMMPS
import math
import matplotlib
##matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import os
import shutil
from numpy import linalg as LA
import sys
from sklearn.cluster import KMeans

##f1='restart_network_0.txt'
##f2='restart_network_1200.txt'
##network_0=np.genfromtxt(f1,skip_header=5021)
####network_1200=np.genfromtxt(f2,skip_header=5021)
##initial_connected_atoms=network_0[:,2:4]
import random
random.seed(a=10)#None, version=2)
##plt.style.use('ggplot')
import param as p

import scipy.optimize as opt

threshold=0.4


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def chain_length_dist(n_chains, chains, links, Lx, Ly, Lz):
##        print(chains[:,1:])
        chain_lengths=[]
##        dist = np.zeros((n_chains,4))
##        dist[:,0:3] = chains[:,1:]
##        dist[:,3] = -1
            
        for i in range (0, n_chains):
            if(chains[i,2] !=-1):
          
              link_1 = chains[i,2]-1
              link_2 = chains[i,3]-1
              lk = links[link_1,:] - links[link_2,:]
              
              lk[0] = lk[0] - int(round(lk[0]/Lx))*Lx
              #lk[1] = lk[1] - int(round(lk[1]/Ly))*Ly
              #lk[2] = lk[2] - int(round(lk[2]/Lz))*Lz
                    
##              dist[i,3] = LA.norm(lk)
              chain_lengths.append(LA.norm(lk))#dist[i,3])
##              meanr2=meanr2+(dist[i,3])**2
    ##          print(dist[i,3])
    ##          stop
    ##          print(((dist[i,3])**2)/(p.N_low*p.b_low**2))
              

        return np.array(chain_lengths)

def chain_length_dist_x(n_chains, chains, links, Lx, Ly, Lz):
##        print(chains[:,1:])
        chain_lengths=[]
##        dist = np.zeros((n_chains,4))
##        dist[:,0:3] = chains[:,1:]
##        dist[:,3] = -1
            
        for i in range (0, n_chains):
            if(chains[i,2] !=-1):
          
              link_1 = chains[i,2]-1
              link_2 = chains[i,3]-1
              lk = links[link_1,:] - links[link_2,:]
              
              lk[0] = lk[0] - int(round(lk[0]/Lx))*Lx
              #lk[1] = lk[1] - int(round(lk[1]/Ly))*Ly
              #lk[2] = lk[2] - int(round(lk[2]/Lz))*Lz
                    
##              dist[i,3] = LA.norm(lk)
              chain_lengths.append(abs(lk[0]))#dist[i,3])
##              meanr2=meanr2+(dist[i,3])**2
    ##          print(dist[i,3])
    ##          stop
    ##          print(((dist[i,3])**2)/(p.N_low*p.b_low**2))
              

        return np.array(chain_lengths)



def force_dist(chain_lengths):
##        print(chains[:,1:])
        force=[]
##        dist = np.zeros((n_chains,4))
##        dist[:,0:3] = chains[:,1:]
##        dist[:,3] = -1
            
        for i in chain_lengths:
            force.append(get_bondforce(i))#dist[i,3])          

        return np.array(force)
##    

def invlangevin(x):
        return x*(2.99942 - 2.57332*x + 0.654805*x**2)/(1-0.894936*x - 0.105064*x**2)

def kuhn_stretch(lam, E_b):
        def func(x, lam, E_b):
            y = lam/x
            beta = invlangevin(y)
            return E_b*np.log(x) - lam*beta/x

        if lam == 0:
           return 1
        else:
           lam_b = opt.root_scalar(func,args=(lam, E_b),bracket=[lam,lam+1],x0=lam+0.05)
           return lam_b.root

        
def get_bondforce(r):

        #K  = p.K
        r0 = 0
        Nb = p.N # b = 1 (lenght scale of the system)
        E_b = p.E_b_low
 
        x = (r-r0)/Nb
        if(x<0.90):
           lam_b = 1.0
           fbkT  = invlangevin(x)
##           fbond = -K*fbkT/r
        elif(x<1.4):
           lam_b = kuhn_stretch(x, E_b)
           fbkT  = invlangevin(x/lam_b)/lam_b
##           fbond = -K*fbkT/r
        else:
           lam_b = x + 0.05
           fbkT  = 325 + 400*(x-1.4)
##           stop
##           fbond = -K*fbkT/r
 
        return fbkT

def readLAMMPS_restart(filename, vflag,G):

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


##   chains[:,0] = N
#cnt,ctype,1,conn1,conn2
   if(vflag==0):
      chains[:,0:4] = np.genfromtxt(filename,usecols=(0,1,2,3), skip_header=17+n_links+3, max_rows=n_chains)
   elif(vflag==1):
      chains[:,0:4] = np.genfromtxt(filename,usecols=(0,1,2,3), skip_header=17+2*n_links+2*3, max_rows=n_chains)
   else:
      print("Invalid Velocity Flag")
##   print(chains)

####   for c in chains:
####      [lnk_1,lnk_2]=c[2:4]
####      G.add_edge(lnk_1,lnk_2)
######   directory = './'+str(int(100*frac_weak))+'/'
####   filename = 'primary_loops'
######   file_path = os.path.join(directory, filename)
######   if not os.path.isdir(directory):
######      os.mkdir(directory)  
####   loop_atoms = np.genfromtxt(filename , usecols=(1), skip_header=0)
####   loop_atoms.tolist() 

   return xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, links, chains, atom_types, bond_types, mass, loop_atoms,G




import networkx as nx

vflag=0
frac_weak_arr=[0.0]#,0.4,0.5]
lams_max=[4]
fig_cnt=4
R=p.b*p.N ## chain end-to-end distance  p.b*np.sqrt(p.N)# end to end distance

##stop


for frac_weak in frac_weak_arr:
           mean_end_to_end_dist=[]
           directory = './function_files/'
           orig_dir = os.path.dirname(directory)
           files=os.listdir(orig_dir)
           directory = './'+str(round(100*frac_weak))+'/'
           if not os.path.isdir(directory):
              os.mkdir(directory)

           for fname in files:
               if(fname!='__pycache__' and fname!='.ipynb_checkpoints'):
             
            # copying the files to the
            # destination directory
                   shutil.copy2(os.path.join(orig_dir,fname), directory)


        # now add path to frac_weak directory
           file_dir = os.path.dirname(directory)
           sys.path.append(file_dir)
           import ioLAMMPS
           import netgen
           from relax import Optimizer

##           lams_max=p.lam_max
           
              
           cnt_lam=1#-3
           for lam_max in lams_max:
               
              
##               [xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, links, chains, atom_types, bond_types, mass, loop_atoms, G]=ioLAMMPS.readLAMMPS_restart("./"+str(round(frac_weak*100))+"/lambda_"+str(int(lam_max))+"/restart_network_0.txt", vflag,frac_weak,lam_max)
               G=nx.Graph()
               [xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, links, chains, atom_types, bond_types, mass, loop_atoms,G]=ioLAMMPS.readLAMMPS("./"+str(round(frac_weak*100))+"/network.txt", vflag,frac_weak,G)
##               dist=np.zeros(n_chains)
##               dist_x=np.zeros(n_chains)
####               count=0
##               for i in range(0, n_chains):
##               ##   if(bond_broken_vs_time[int(i)]==len(time_step_list)):
##                  if(True):
##                     j1=chains[i,2]-1#atoms 1 - one end of chain # -1 done because this is an index
##                     j2=chains[i,3]-1#atom 2- another end of chain
##                  ##   for temp in range(0,len(atoms)):
##                  ##      if(atoms[
##                     t1=links[j1,:] #atom 1 coordinates
##                     t2=links[j2,:] #atom 2 coordinates
##                     dist[i]=np.sqrt((t1[0]-t2[0])**2+(t1[1]-t2[1])**2+(t1[1]-t2[1])**2)
##                     dist_x[i]=abs(t1[0]-t2[0])
##                     if(dist_x[i]<1e-12):
##                        stop
               Lx=xhi-xlo
               Ly=(yhi-ylo)*10  # to make sure that the distances are not being calculated across the PBC
               Lz=(zhi-zlo)*10
               dist=chain_length_dist(n_chains, chains, links, Lx, Ly, Lz)
##               print('min(dist_x)',min(dist_x))
               print('min(dist)',min(dist))
               a=np.histogram(dist)
               a=np.histogram(dist,100)
               plt.figure(1+fig_cnt)
##               plt.plot(a[1][0:-1],a[0],'o-',label='init')
               plt.legend()
               print('init: a[1][0]',a[1][1])
               plt.xlabel('chain extension=chain length normalized by end to end distance of gaussian chain ')
               plt.ylabel('frequency (normalized by number of considered chains)')## (normalized by number of chains considered for histogram (all chains except in initial bin)')
               


               dist=force_dist(dist)
    ##               print('min(dist_x)',min(dist_x))
               print('min(dist)',min(dist))
               a=np.histogram(dist)
               a=np.histogram(dist,100)
##                   R=random.randrange(0,1)
##                   G=random.randrange(0,1)
##                   B=random.randrange(0,1)
##                   RGB=np.random.random((3,1))#(range(0,1), 3)
               plt.figure(2+fig_cnt)
               plt.xlabel('force on chain (fb/kBT)' )##(KBT.nm^(-1))')
               plt.ylabel('frequency (normalized by number of considered chains)')## (normalized by number of chains considered for histogram (all chains except in initial bin)')
##               plt.plot(a[1][1:-1],a[0][1:],'o-',label='init')#,color=(r[cnt],g[cnt],b[cnt]))

               
               
               print('sum(a[0]) total number of chains' ,sum(a[0]))
##               plt.show()
               plt.legend()
               # plt.xlabel('Force on chain')
               # plt.ylabel('Frequency')
##               ite_arr=[10,50,150,200,380]#400,500,600,700]#,1200,1500,1560]#[10,20,30,50,70,100,150,250,380,386]
##               ite_arr=np.arange(10,n_chains*0.8,50,dtype='int')
               transition_ite=int((p.lam_step-1)/(p.e_rate*p.del_t))
               broken_data=np.genfromtxt('t_KMC_and_ite_failure.txt')
               ite_broken=broken_data[1]
               ite_broken_rounded=int(p.wrt_step*int(ite_broken/p.wrt_step))
               ite_arr=np.append(np.array([0]),np.arange(transition_ite,692,max(2*p.wrt_step,5),dtype='int'))
##               ite_app=np.append(ite_arr,0)
               r=np.linspace(0.1,0.9,len(ite_arr))
               g=np.linspace(0.3,0.7,len(ite_arr))
               b=np.linspace(0.1,0.9,len(ite_arr))
               cnt=-1
               max_force_overall=np.zeros(len(ite_arr))
               width=np.zeros(len(ite_arr))
               gini_chain_length=np.zeros(len(ite_arr))
               gini_force=np.zeros(len(ite_arr))
               ##stop
               
               for ite in ite_arr:
                   print('ite',ite)
                   cnt=cnt+1
                   [xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, links, chains, atom_types, bond_types, mass, loop_atoms, G]=readLAMMPS_restart("./"+str(round(frac_weak*100))+"/restart_network_"+str(ite)+".txt", vflag,frac_weak)
                   ##print('n_chains',n_chains)
    ##               [xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, links, chains, atom_types, bond_types, mass, loop_atoms, G]=ioLAMMPS.readLAMMPS("./"+str(round(frac_weak*100))+"/lambda_"+str(int(lam_max))+"/network.txt", vflag,frac_weak, lam_max)
    ##               dist=np.zeros(n_chains)
    ##               dist_x=np.zeros(n_chains)
    ####               count=0
    ##               for i in range(0, n_chains):
    ##               ##   if(bond_broken_vs_time[int(i)]==len(time_step_list)):
    ##                  if(True):
    ##                     j1=chains[i,2]-1#atoms 1 - one end of chain # -1 done because this is an index
    ##                     j2=chains[i,3]-1#atom 2- another end of chain
    ##                  ##   for temp in range(0,len(atoms)):
    ##                  ##      if(atoms[
    ##                     t1=links[j1,:] #atom 1 coordinates
    ##                     t2=links[j2,:] #atom 2 coordinates
    ##                     dist[i]=np.sqrt((t1[0]-t2[0])**2+(t1[1]-t2[1])**2+(t1[1]-t2[1])**2)
    ##                     dist_x[i]=abs(t1[0]-t2[0])
    ##                     if(dist_x[i]<1e-12):
    ##                        stop
                   Lx=xhi-xlo
                   Ly=(yhi-ylo)*10  # to make sure that the distances are not being calculated across the PBC
                   Lz=(zhi-zlo)*10
##                   stop
                   dist=chain_length_dist(n_chains, chains, links, Lx, Ly, Lz)
                   ##print('chain length arr',dist)

                   mean_end_to_end_dist.append(np.mean(dist))
    ##               print('min(dist_x)',min(dist_x))
                   print('min(dist)',min(dist))
                   ##a=np.histogram(dist)
                   a=np.histogram(dist,100)
##                   R=random.randrange(0,1)
##                   G=random.randrange(0,1)
##                   B=random.randrange(0,1)
##                   RGB=np.random.random((3,1))#(range(0,1), 3)
                   
                   plt.figure(1+fig_cnt)
                   y=a[0][1:]/sum(a[0][1:])
                   x=a[1][1:-1]/R
                   plt.plot(x,y,'o-',label=str(ite),color=(r[cnt],g[cnt],b[cnt]))
                   
                   ##stop
                   
                   np.savetxt("./"+str(round(frac_weak*100))+"/chain_length_dist"+str(int(100*frac_weak))+'ite_'+str(ite)+'.txt', np.transpose(np.array([list(x),list(y)])))
##                   stop
                   print('a[1][1]',a[1][1])
                   print('number of chains considered',sum(a[0][1:]))
                   print('total number of chains',sum(a[0][0:]))
                   gini_chain_length[cnt]=gini(dist)
##                   plt.show()

                   dist_x=chain_length_dist_x(n_chains, chains, links, Lx, Ly, Lz)
    ##               print('min(dist_x)',min(dist_x))
##                   print('min(dist)',min(dist))
##                   a=np.histogram(dist_)
                   a=np.histogram(dist_x,100)
    ##               plt.plot(a[1][0:-1],a[0],'o-',label='init')
                   plt.legend()
                   print('a[1][0]',a[1][1])
                   
                   
                   delta_x_min=a[1][1]
                   deltax=delta_x_min*0.9  # the delta_x considered for the binning should be less than the minimum chain end to end distance of chains in the network
                   if(deltax<1e-5):
                           continue


                   dist=force_dist(dist)
                   ##print('force arr',dist)
                   max_force_overall[cnt]=np.max(dist)
    ##               print('min(dist_x)',min(dist_x))
                   print('min(dist)',min(dist))
                   a=np.histogram(dist)
                   a=np.histogram(dist,100)
##                   R=random.randrange(0,1)
##                   G=random.randrange(0,1)
##                   B=random.randrange(0,1)
##                   RGB=np.random.random((3,1))#(range(0,1), 3)
                   plt.figure(2+fig_cnt)
                   y=(a[0][1:])/sum(a[0][1:])
                   x=a[1][1:-1] ## (fb/kBT)' )##  force in KBT.nm^-1)
                   plt.plot(x,y,'o-',label=str(ite),color=(r[cnt],g[cnt],b[cnt]))
                   ##y=(a[0][1:])/n_chains
                   ##stop
                   np.savetxt("./"+str(round(frac_weak*100))+"/force_dist"+str(int(100*frac_weak))+'ite_'+str(ite)+'.txt', np.transpose(np.array([list(x),list(y)])))
                   gini_force[cnt]=gini(dist)
                   
                   ##stop
                
                
                   delta_x_plot=deltax*2
                   density=np.zeros(math.floor(int((xhi-xlo)/delta_x_plot))+1)  ## density in each bin=fraction of total number of chains passing through each bin
                   ##density_weak=np.zeros(math.floor(int((xhi-xlo)/delta_x_plot))+1)
                   ##density_strong=np.zeros(math.floor(int((xhi-xlo)/delta_x_plot))+1)
                   mean_force=np.zeros(math.floor(int((xhi-xlo)/delta_x_plot))+1)
                   ##mean_force_weak=np.zeros(math.floor(int((xhi-xlo)/delta_x_plot))+1)
                   ##mean_force_strong=np.zeros(math.floor(int((xhi-xlo)/delta_x_plot))+1)
##                   max_force=np.zeros(math.floor(int((xhi-xlo)/deltax))+1)
                   mean_extension=np.zeros(math.floor(int((xhi-xlo)/delta_x_plot))+1)

                   x_count=-1 # we are pulling in x direction, not z
                   xrange=np.zeros(len(density))#[z for z in range(zlo,zhi,deltax)]
                   x2_pos=xlo
                   # tensor=np.zeros((n_chains,math.floor(int((xhi-xlo)/delta_x_plot))))
                   for temp in range(0,math.floor(int((xhi-xlo)/delta_x_plot))):#int(zlo),int(zhi),deltax):
                    ## for every bin- go through all the chains and check which ones fall into the bin
                    ## this is the chain density=number of chains crossing thorugh every bin- the sum of density might be different for different iterations based on the clustering, etc
                       x_count=x_count+1
                       forces=[] # stores the values of forces (fb/kBT)' )##(kBT*(nm^-1)) on all chains passign through this x plane/bin
                       ##forces_weak=[]
                       ##forces_strong=[]
                       extensions=[]
                    ##   z=z
                    ##   z=xrange
                       x1=x2_pos
                       x2_pos=x1+delta_x_plot
                       x2=x1+deltax
                    ##   print('x2',x2)
                    ##   if(x2>(xhi-xlo)):
                    ##      x2=0 # periodic boundary condition
                    ##      x_count=0
                       xrange[x_count]=x2
                    ##   print('xcount',x_count)
                       img_cnt=0
                       for i in range(0, n_chains):
                    ##      if(bond_broken_vs_time[int(i)]==len(time_step_list)):
                          if(True):
                             j1=chains[i,2]-1#atoms 1 - one end of chain
                             j2=chains[i,3]-1#atom 2- another end of chain
                             ctype=chains[i,0]
                          ##   for temp in range(0,len(atoms)):
                          ##      if(atoms[
                             t1=links[j1,0] #atoms 1 x-coordinates
                             t2=links[j2,0] #atom 2 x-coordinates
##                             avg_x=(t1+t2)/2.0

                             # THIS IS DOEN ASSUMING THAT THE CROSSLINKERS 1 AND 2 ARE ORDERED 
                             image=False

                             length=abs(t1-t2)
##                             if(t1>t2):
####                                     t1=t1- Lx# cnsider the image
##                                     length=(t1-t2)-Lx
##                                     image=True
##                                     
####                                     x1_new=
####                                     stop
##                             else:
##                                     length=(t1-t2)
                             
                             
                             chain_length= abs(length - int(round(length/Lx))*Lx)
                             if(length!=chain_length):
                                     image=True
##                             if((avg_x<xlo) or (avg_x>xhi)): # this case will never arise - because periodic bc already implememted
##                                density[0]=density[0]+1
##                                if(ctype==1):
##                                   density_weak[0]=density_weak[0]+1#(t1-x1)/deltax
##                                elif(ctype==2):
##                                   density_strong[0]=density_strong[0]+1#(t1-x1)/deltax
                             number_of_bins=int(chain_length/delta_x_plot)+1
                             assigned=False
                             if(True):##chain_length>deltax):
                                if(image==False and (t1-x1)*(t2-x2)<0):#(avg_x-x1)*(avg_x-x2)<0):
                                 # chain length is greater than bin length and the chain ends are on opposite sides of the bin ends(meaning that the chain crosses the bin)
                                        density[x_count]=density[x_count]+1  ##1/number_of_bins
                                        # tensor[i,x_count]=tensor[i,x_count]+1
                                        assigned=True
                                        
                                        f=get_bondforce(chain_length)
                                        forces.append(f)
                                        extensions.append(chain_length/R)
#                                         if(ctype==0):
#                                                 forces_weak.append(f)
                                           
#                                         elif(ctype==1):
#                                            forces_strong.append(f)  
                                elif(image==True and (t1-x1)*(t2-x2)>0):#(avg_x-x1)*(avg_x-x2)<0):
                                        img_cnt=img_cnt+1
                                 # chain length is greater than bin length and the chain ends are on same sides of the bin ends, but this is through the image(meaning that the chain crosses the bin)
                                        density[x_count]=density[x_count]+1 ##1/number_of_bins
                                        f=get_bondforce(chain_length)
                                        forces.append(f)
                                        extensions.append(chain_length/R)
                                        # tensor[i,x_count]=tensor[i,x_count]+1
                                        assigned=True
                                        
                             if(assigned==False):##chain_length<deltax):
                                if(image==False and ((t1-x1)*(t2-x1))*((t1-x2)*(t2-x2))<0):#(avg_x-x1)*(avg_x-x2)<0):
                         # chain length is greater than bin length and the chain ends are on opposite sides of the bin ends(meaning that the chain crosses the bin)
                                    density[x_count]=density[x_count]+1  ##1/number_of_bins
                                    # tensor[i,x_count]=tensor[i,x_count]+1
                                    assigned=True   
                                    f=get_bondforce(chain_length)
                                    forces.append(f)
                                    extensions.append(chain_length/R)
#                                         if(ctype==0):
#                                                 forces_weak.append(f)
                                           
#                                         elif(ctype==1):
#                                            forces_strong.append(f)                                        

                       mean_force[x_count]=np.mean(forces)
                       mean_extension[x_count]=np.mean(extensions)
##                       max_force[x_count]=(np.max(forces))
                       # try:
                       #         mean_force_weak[x_count]=np.mean(forces_weak)
                       # except:
                       #          mean_force_weak[x_count]=0
                       # mean_force_strong[x_count]=np.mean(forces_strong)
##                       print('img_cnt',img_cnt)
                   # stop
#                    for i in range(0,n_chains):

#                     if(np.sum(tensor[i,:])!=0):
#                         tensor[i,:]=tensor[i,:]/np.sum(tensor[i,:])
#                     ##tensor[i,:]=tensor[i,:]/np.sum(tensor[i,:])
#                     ##print('bond idx=',i,'sum=',np.sum(tensor[i,:]))
#                     new_x=np.zeros(np.shape(density))
#                     #print('np.where(np.sum(tensor[i,:]))==0',np.where(np.sum(tensor[i,:])==0))

#                     for cnt_t in range(0,math.floor(int((xhi-xlo)/delta_x_plot))):
#                         new_x[cnt_t]=np.sum(tensor[:,cnt_t])
#                     intg_new=np.sum(new_x)
                    # print('np.sum(new_x)',np.sum(new_x))
                    #print('np.shape(np.where(np.sum(tensor[:],axis=1)==0))',np.shape(np.where(np.sum(tensor[:],axis=1)==0)))


                    # x=new_x
                    # density_all_y[cnt,run_cnt,:]=x
            
            
                   # stop
                   plt.figure(3+fig_cnt)
                   x=(xrange[0:-1]-xlo)/(xhi-xlo) ## distance along tensile axis - normalized by total length along tensile axis
                   y= sum(a[0][1:])*density[0:-1]/sum(density[0:-1])#/sum(a[0][1:])
                   plt.plot(x,y,'o-',color=(r[cnt],g[cnt],b[cnt]),label=str(ite))#(RGB[0][0],RGB[1][0],RGB[2][0]))
                   
                   plt.legend()
                   np.savetxt("./"+str(round(frac_weak*100))+"/chain_denisty_along_axis_ite_"+str(ite)+".txt",np.transpose(np.array([list(x), list(y)])))
                   ##stop
                                                                                                                                     
##                   plt.title('weak_Frac='+str(frac_weak))
                   ## getting the fracture zone width

                   
                   a=np.where(density[0:-1]<threshold*(np.max(density[0:-1])))#-np.min(density[0:-1]))+np.min(density[0:-1]))
                   a_copy=a[0].copy()

                        
                   if(len(a[0])>2): # at least 3 datapoints required                         
                           kmeans = KMeans(n_clusters=2)
                           kmeans.fit(a_copy.reshape(-1, 1))
                           clusters = kmeans.predict(a_copy.reshape(-1, 1))
                           crack1=np.where(clusters==0)
                           crack2=np.where(clusters==1)
                           crack1_1=xrange[a[0][crack1[0][0]]]
                           crack1_2=xrange[a[0][crack1[0][-1]]]


                           crack2_1=xrange[a[0][crack2[0][0]]]
                           crack2_2=xrange[a[0][crack2[0][-1]]]

                        
                           width[cnt]=((crack1_2-crack1_1)+(crack2_2-crack2_1))/(xhi-xlo)
                   else:
                           width[cnt]=0  
                        
                   
                   print('width[cnt] (normalized by length of tensile axis)',width[cnt])
                   # stop

                   
#                    plt.figure(5+fig_cnt)

#                    plt.plot(xrange[0:-1], mean_force_weak[0:-1],'o-',color=(r[cnt],g[cnt],b[cnt]),label=str(ite))#(RGB[0][0],RGB[1][0],RGB[2][0]))
#                    plt.title('weak_Frac='+str(frac_weak))
                   
#                    plt.figure(6+fig_cnt)

#                    plt.plot(xrange[0:-1], mean_force_strong[0:-1],'o-',color=(r[cnt],g[cnt],b[cnt]),label=str(ite))#(RGB[0][0],RGB[1][0],RGB[2][0]))
                   
                   plt.figure(4+fig_cnt)

                   plt.plot(x, mean_force[0:-1],'o-',color=(r[cnt],g[cnt],b[cnt]),label=str(ite))
                   
                   np.savetxt("./"+str(round(frac_weak*100))+"/mean_force_along_axis_ite_"+str(ite)+".txt",np.transpose(np.array([list(x), list(mean_force[0:-1])])))
                   np.savetxt("./"+str(round(frac_weak*100))+"/mean_extension_along_axis_ite_"+str(ite)+".txt",np.transpose(np.array([list(x), list(mean_extension[0:-1])])))

                   print('np.sum(density)',np.sum(density))
                   ##print('n_chains',n_chains)
               np.savetxt("./"+str(round(frac_weak*100))+"/mean_end_to_end_distance_normtxt",np.transpose(np.array([ite_arr, list(np.array(mean_end_to_end_dist)/R)])), header="ite_arr, mean_end_to_end_norm_by_Nb")

               plt.figure(2+fig_cnt)
               plt.title('weak_Frac='+str(frac_weak))
               plt.legend()
               ##plt.xlabel('Force on chain')
               ##plt.ylabel('Frequency')

               plt.savefig("./"+str(round(frac_weak*100))+"/force_distribution.png")

               
               plt.figure(1+fig_cnt)
               plt.title('weak_Frac='+str(frac_weak))
               plt.legend()
               ##plt.xlabel('Chain length')
               ##plt.ylabel('Frequency')
               plt.savefig("./"+str(round(frac_weak*100))+"/chain_length_distribution.png")

               
               plt.figure(3+fig_cnt)
               plt.title('weak_Frac='+str(frac_weak))
               plt.legend()
               plt.xlabel('distance along tensile axis - normalized by total length along tensile axis')
               plt.ylabel('Chain density (number of chains through this bin/total number of chains through each bin)')
               plt.savefig("./"+str(round(frac_weak*100))+"/chain_density.png")
               

               
               plt.figure(4+fig_cnt)
               plt.title('weak_Frac='+str(frac_weak))
               plt.legend(ite_arr,)
               plt.xlabel('length along tensile axis (normalized by axis length)')
               plt.ylabel('mean force (fb/kBT)' )##(kBT.nm^-1)')
               plt.savefig("./"+str(round(frac_weak*100))+"/mean_force_chains.png")


               plt.figure(5+fig_cnt)
               plt.title('weak_Frac='+str(frac_weak))
               plt.legend()
               plt.xlabel('Iteration')
               plt.ylabel('Mean end-to-end distance (norm by Nb)')
               plt.plot(ite_arr,list(np.array(mean_end_to_end_dist)/R),'o-')
               plt.axvline(x=transition_ite, color='r', linestyle='--', linewidth=2)
               plt.savefig("./"+str(round(frac_weak*100))+"/mean_force_weak_chains.png")

               plt.figure(6+fig_cnt)
               # plt.title('weak_Frac='+str(frac_weak))
               # plt.legend()
               # plt.xlabel('x position')
               # plt.ylabel('Average of forces on STRONG chains passing through x plane')
               # plt.savefig("./"+str(round(frac_weak*100))+"/mean_force_strong_chains.png")

               plt.figure(1)
##               plt.title('weak_Frac='+str(frac_weak))
               plt.plot()
               plt.plot(ite_arr,max_force_overall,'o-',label='frac_weak='+str(frac_weak))
               plt.axvline(x=transition_ite, color='r', linestyle='--', linewidth=2)
               plt.xlabel('ite')
               plt.ylabel('Max force among chains in network')
               plt.legend()

               plt.figure(2)
               ##width=width/R
               plt.plot(ite_arr,width,'o-',label='frac_weak='+str(frac_weak))
               plt.axvline(x=transition_ite, color='r', linestyle='--', linewidth=2)
               plt.xlabel('ite')
               plt.ylabel('Crack width (normalized by box length)')
               plt.legend()
               plt.savefig('crack_width_ite'+str(int(100*frac_weak)))
               np.savetxt('crack_width_ite'+str(int(100*frac_weak))+".txt",np.transpose(np.array([ite_arr,width])))


               plt.figure(3)
               plt.plot(ite_arr,gini_chain_length,'o-',label='chain_length_frac_weak='+str(frac_weak))
               plt.axvline(x=transition_ite, color='r', linestyle='--', linewidth=2)
               plt.xlabel('ite')
               plt.ylabel('gini_chain_length')
               plt.legend()
               plt.savefig('gini_chain_length')
            
               ##plt.savetxt()

##               plt.figure(4)
               plt.plot(ite_arr,gini_force,'o-',label='force_frac_weak='+str(frac_weak))
               plt.axvline(x=transition_ite, color='r', linestyle='--', linewidth=2)
               plt.xlabel('ite')
               plt.ylabel('gini_coefficienth')
               plt.legend()
               plt.savefig('gini_chainlength_and_force'+str(int(100*frac_weak)))

               plt.figure(4)
               # plt.plot(ite_arr,gini_force,'o-',label='frac_weak='+str(frac_weak))
               # plt.xlabel('ite')
               # plt.ylabel('gini_force')
               # plt.legend()
               # plt.savefig('gini_force'+str(int(100*frac_weak)))
               np.savetxt("./"+str(round(frac_weak*100))+"/gini_coeff.txt", np.transpose(np.array([ite_arr, gini_chain_length, gini_force])), header="ite_arr, gini_chain_length, gini_force")


               
               fig_cnt=fig_cnt+6
plt.show()


