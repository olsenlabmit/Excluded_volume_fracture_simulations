import numpy as np
##

import math
import matplotlib
##matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import param as p
import os
import shutil
import sys
sys.path.append('./function_files/')
import ioLAMMPS

##directory = './function_files/'
##orig_dir = os.path.dirname(directory)
##files=os.listdir(orig_dir)
##
##
####for fname in files:
####   if(fname=='ioLAMMPS.py'):
#### # copying the files to the
#### # destination directory
####       shutil.copy2(os.path.join(orig_dir,fname), directory)
##
##import function_files/ioLAMMPS.py as ioLAMMPS

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

   return xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, links, chains, atom_types, bond_types, mass, loop_atoms






vflag=0
frac_weak=0.0

broken_data=np.genfromtxt('t_KMC_and_ite_failure_step_plus_cont.txt')
ite_broken=broken_data[1]
ite_broken_rounded=int(p.wrt_step*int(ite_broken/p.wrt_step))
ite_init=0
transition_ite=int((p.lam_max-1)/(p.e_rate*p.del_t))
ite_init_step=transition_ite  # init ite for just step strain sim

f1='./'+str(int(100*frac_weak))+'/'+'restart_network_'+str(ite_init)+'.txt'  # init box
f2='./'+str(int(100*frac_weak))+'/'+'restart_network_'+str(ite_broken_rounded)+'.txt'   # final box
f3='./'+str(int(100*frac_weak))+'/'+'restart_network_'+str(ite_init_step)+'.txt'   # init box for step strain (end of continuous sim)



[xlo_final, xhi_final, ylo_final, yhi_final, zlo_final, zhi_final, n_links, n_chains_final, links_final, chains_final, atom_types, bond_types, mass, loop_atoms_final]=readLAMMPS_restart(f2, vflag,frac_weak)
[xlo_init, xhi_init, ylo_init, yhi_init, zlo_init, zhi_init, n_links, n_chains_init, links_init, chains_init, atom_types, bond_types, mass, loop_atoms]=readLAMMPS_restart(f1, vflag,frac_weak)

[xlo_init_step, xhi_init_step, ylo_init_step, yhi_init_step, zlo_init_step, zhi_init_step, n_links_step, n_chains_init_step, links_init_step, chains_init_step, atom_types_step, bond_types_step, mass_step, loop_atoms_step]=readLAMMPS_restart(f3, vflag,frac_weak)

n_chains_broken=-n_chains_final+n_chains_init
chains_broken=np.zeros((n_chains_broken,4),dtype='int')
print('n_chains_broken- overall', n_chains_broken)
cnt=0
found_cnt=0
i_not_found_arr=[i for i in range(n_chains_final)]
for c in chains_init:
   lnk_1=c[2]
   lnk_2=c[3]
   found=False

   for i in i_not_found_arr:##range(n_chains_final):
##      if(i not in i_found_arr):
         if((chains_final[i,2]==lnk_1 and chains_final[i,3]==lnk_2)):# or (chains_final[i,2]==lnk_2 and chains_final[i,3]==lnk_1)): # means that chain is present
            found=True
##            print(lnk_1,lnk_2)
            found_cnt=found_cnt+1
            i_not_found_arr.remove(i)
            break
   if(found==False): # chain is not found in the final array
         
         chains_broken[cnt,:]=c
         cnt=cnt+1
##         break
         
         
ioLAMMPS.writeLAMMPS('./'+str(int(100*frac_weak))+'/'+'broken_chains_init_box.txt',xlo_init, xhi_init, ylo_init, yhi_init, zlo_init, zhi_init, links_init,chains_broken, atom_types, bond_types, mass, loop_atoms_final)

ioLAMMPS.writeLAMMPS('./'+str(int(100*frac_weak))+'/'+'broken_chains_final_box.txt',xlo_final, xhi_final, ylo_final, yhi_final, zlo_final, zhi_final, links_final,chains_broken, atom_types, bond_types, mass, loop_atoms_final)

ioLAMMPS.writeLAMMPS('./'+str(int(100*frac_weak))+'/'+'broken_chains_init_step_box_all.txt',xlo_init_step, xhi_init_step, ylo_init_step, yhi_init_step, zlo_init_step, zhi_init_step, links_init_step,chains_broken, atom_types, bond_types, mass, loop_atoms_final)


# now track the chains broken only during the step hold simulation
n_chains_broken=-n_chains_final+n_chains_init_step
chains_broken=np.zeros((n_chains_broken,4),dtype='int')
print('n_chains_broken- in step hold part', n_chains_broken)
cnt=0
found_cnt=0
i_not_found_arr=[i for i in range(n_chains_final)]
for c in chains_init_step:
   lnk_1=c[2]
   lnk_2=c[3]
   found=False

   for i in i_not_found_arr:##range(n_chains_final):
##      if(i not in i_found_arr):
         if((chains_final[i,2]==lnk_1 and chains_final[i,3]==lnk_2)):# or (chains_final[i,2]==lnk_2 and chains_final[i,3]==lnk_1)): # means that chain is present
            found=True
##            print(lnk_1,lnk_2)
            found_cnt=found_cnt+1
            i_not_found_arr.remove(i)
            break
   if(found==False): # chain is not found in the final array
         
         chains_broken[cnt,:]=c
         cnt=cnt+1
##         break

         
ioLAMMPS.writeLAMMPS('./'+str(int(100*frac_weak))+'/'+'broken_chains_init_step_box.txt',xlo_init_step, xhi_init_step, ylo_init_step, yhi_init_step, zlo_init_step, zhi_init_step, links_init_step,chains_broken, atom_types, bond_types, mass, loop_atoms_final)
# this contains the chains broken only during the step hold simulation, as translated back to the starting box of the step hold sim (=ending box of the continuous sim)



n_chains_broken=-n_chains_init_step+n_chains_init  # now track the chains broken only during the first continuous step
chains_broken=np.zeros((n_chains_broken,4),dtype='int')
print('n_chains_broken- overall', n_chains_broken)
cnt=0
found_cnt=0
i_not_found_arr=[i for i in range(n_chains_init_step)]
for c in chains_init:
   lnk_1=c[2]
   lnk_2=c[3]
   found=False

   for i in i_not_found_arr:##range(n_chains_final):
##      if(i not in i_found_arr):
         if((chains_init_step[i,2]==lnk_1 and chains_init_step[i,3]==lnk_2)):# or (chains_final[i,2]==lnk_2 and chains_final[i,3]==lnk_1)): # means that chain is present
            found=True
##            print(lnk_1,lnk_2)
            found_cnt=found_cnt+1
            i_not_found_arr.remove(i)
            break
   if(found==False): # chain is not found in the final array
         
         chains_broken[cnt,:]=c
         cnt=cnt+1
##         break

         

ioLAMMPS.writeLAMMPS('./'+str(int(100*frac_weak))+'/'+'broken_chains_cont_only_init_box.txt',xlo_init, xhi_init, ylo_init, yhi_init, zlo_init, zhi_init, links_init,chains_broken, atom_types, bond_types, mass, loop_atoms_final)





stop


network_0=np.genfromtxt(f1,skip_header=5021)
##network_1200=np.genfromtxt(f2,skip_header=5021)
initial_connected_atoms=network_0[:,2:4]
##np.savetxt('test.txt',initial_connected_atoms,header='## ')
vflag=0
frac_weak=0.3
bond_broken_vs_time=np.zeros(len(initial_connected_atoms))# contains the time step during which the bond has broken

time_count=0;
time_step_list=[190]
for time_step in time_step_list:#,1000,1200]:
   index=[] # keeps track of which bonds are intact (not broken) at final (or intermediately before) time step
   bond_broken_in_this_step=0
   time_count=time_count+1
   fnew='restart_network_'+str(time_step)+'.txt'
   print(fnew)
   network_new=np.genfromtxt(fnew,skip_header=5021)
   final_connected_atoms=network_new[:,2:4]

   for i in range(0, len(initial_connected_atoms)): # going over each bond
##      not_broken=0
      atom1=initial_connected_atoms[i,0]
      atom2=initial_connected_atoms[i,1]
      for j in range(0, len(final_connected_atoms)):
         if(atom1==final_connected_atoms[j,0] and atom2==final_connected_atoms[j,1]):# and ([atom2, atom1] not in final_connected_atoms)):
         #means not broken
            index.append(i)
##            not_broken=1
            break
   for i in range(0, len(initial_connected_atoms)):
      if(i not in index): # means broken
         if(bond_broken_vs_time[i]==0):
            bond_broken_vs_time[i]=time_count
##            print(time_count)
            bond_broken_in_this_step=bond_broken_in_this_step+1
         
   print('time_count',time_count)
   print('bond_broken_in_this_step',bond_broken_in_this_step)
   print(len(index))   
vflag=0
frac_weak=0.3
last_sim_step=190
step_broken=1200
new_step=step_broken
[xlo_final, xhi_final, ylo, yhi, zlo, zhi, n_links, n_chains, links_final, chains_final, atom_types, bond_types, mass, loop_atoms]=ioLAMMPS_new.readLAMMPS_post_run('restart_network_'+str(step_broken)+'.txt', vflag,frac_weak)
final_step_found=0
step_delta=10
n_chains_end=n_chains

while(final_step_found==0):
   new_step=new_step-step_delta

   [xlo_final, xhi_final, ylo, yhi, zlo, zhi, n_links, n_chains, links_final, chains_final, atom_types, bond_types, mass, loop_atoms]=ioLAMMPS_new.readLAMMPS_post_run('restart_network_'+str(new_step)+'.txt', vflag,frac_weak)
   if(n_chains>n_chains_end):
      step_broken=new_step
      final_step_found=1
      break
step_broken_1=step_broken # this is the step at which there is change in the fraction of chains in network- ie. the last step at which at least some of the network is intact
print('step_broken 1',step_broken_1)

final_step_found=0
step_broken=0
new_step=step_broken

[xlo_final, xhi_final, ylo, yhi, zlo, zhi, n_links, n_chains, links_final, chains_final, atom_types, bond_types, mass, loop_atoms]=ioLAMMPS_new.readLAMMPS_post_run('restart_network_'+str(step_broken)+'.txt', vflag,frac_weak)
n_chains_end=n_chains
delta_chains_broken=55# chains broken in a single step
while(final_step_found==0 and new_step<last_sim_step):
   new_step=new_step+step_delta

   [xlo_final, xhi_final, ylo, yhi, zlo, zhi, n_links, n_chains, links_final, chains_final, atom_types, bond_types, mass, loop_atoms]=ioLAMMPS_new.readLAMMPS_post_run('restart_network_'+str(new_step)+'.txt', vflag,frac_weak)
##   print('n_chains_end-n_chains',n_chains_end-n_chains)
   if(n_chains_end-n_chains>=delta_chains_broken): # if in any step the number of chains broken is greater than the number specified
      # means that there are a lot of bonds breaking in this step
      step_broken=new_step
      final_step_found=1
      break
   n_chains_end=n_chains

step_broken_2=step_broken
print('step_broken 2',step_broken_2)


if(step_broken_2>step_broken_1 ):
   step_broken=step_broken_1
elif(step_broken_2>0):
   step_broken=step_broken_2
else:
   step_broken=step_broken_1
##step_broken=520
print('original step broken',step_broken)




def calc_broken_density(step_broken):
      [xlo_final, xhi_final, ylo, yhi, zlo, zhi, n_links, n_chains, links_final, chains_final, atom_types, bond_types, mass, loop_atoms]=ioLAMMPS_new.readLAMMPS_post_run('restart_network_'+str(step_broken)+'.txt', vflag,frac_weak)
      [xlo_init, xhi_init, ylo, yhi, zlo, zhi, n_links, n_chains, links_init, chains_init, atom_types, bond_types, mass, loop_atoms]=ioLAMMPS_new.readLAMMPS_post_run('restart_network_0.txt', vflag,frac_weak)
      xlo=xlo_final
      xhi=xhi_final
      links=links_final
      chains=chains_init
      # for the full list of bond positions- I need the data from restart_network_0
      # but for the atom positions, I need data from the final network file (or the one just afetr fracture)
      x_L=xhi_final-xlo_final
      cnt_length_new=12.0*x_L/(xhi_init-xlo_init)
      ##stop
      bond_types=11#len(index)
      ##stop
      ioLAMMPS_new.writeLAMMPS_post_run_with_time('network_labelled_test_30_nov.txt', xlo, xhi, ylo, yhi, zlo, zhi, links, chains, atom_types, bond_types, mass, loop_atoms,index,bond_broken_vs_time)
      ##stop
      ##for z in range(zlo, zhi, deltax):
      dist=np.zeros(len(index))
      dist_x=np.zeros(len(index))

      deltax=0.2*(xhi_init-xlo_init)#6.0#deltax*x_L/30.0
      density=np.zeros(math.floor(int((x_L)/deltax))+1)
      density_weak=np.zeros(math.floor(int((x_L)/deltax))+1)
      density_strong=np.zeros(math.floor(int((x_L)/deltax))+1)
      density_all_bonds=np.zeros(math.floor(int((x_L)/deltax))+1)
      density_all_bonds_weak=np.zeros(math.floor(int((x_L)/deltax))+1)
      density_all_bonds_strong=np.zeros(math.floor(int((x_L)/deltax))+1)
      ##stop
      x_count=0 # we are pulling in x direction, not z
      xrange=np.zeros(len(density))#[z for z in range(zlo,zhi,deltax)]
      x2=xlo
      max_chain_length=0
      min_chain_length=100
      broken_count=0
      all_bonds_count=0
      for temp in range(0,math.floor(int((xhi-xlo)/deltax))):#int(zlo),int(zhi),deltax):
         x_count=x_count+1
      ##   z=z
      ##   z=xrange
         x1=x2 # x-coordinate of bin 
         x2=x1+deltax

         xrange[x_count]=x2

         for i in range(0, len(initial_connected_atoms)):

            if(i not in index): # broken
               
               j1=chains_init[i,1]-1#atoms 1 - one end of chain
               j2=chains_init[i,2]-1#atom 2- another end of chain
               type=chains_init[i,0]

               t1=links_final[j1,0] #atoms 1 x-coordinates # this coordinate has been taken from the most recent file read using readLAMMPS
               t2=links_final[j2,0] #atom 2 x-coordinates
      ##         avg_x=(t1+t2)/2.0
               chain_length=np.sqrt((links[j1,0]-links[j2,0])**2+(links[j1,1]-links[j2,1])**2+(links[j1,2]-links[j2,2])**2)

               if(chain_length>cnt_length_new):

                  if((t1+t2-x_L)/2.0>xlo):
                     avg_x=(t1+t2-x_L)/2.0
                  if((t1+t2+x_L)/2.0<xhi):
                     avg_x=(t1+t2+x_L)/2.0
                
               else:
                  avg_x=(t1+t2)/2.0

               if((avg_x<xlo) or (avg_x>xhi)): # this case will never arise - because periodic bc already implememted
                  density[0]=density[0]+1
                  broken_count=broken_count+1
                  stop
                  if(type==1):
                     density_weak[0]=density_weak[0]+1#(t1-x1)/deltax
                  elif(type==2):
                     density_strong[0]=density_strong[0]+1#(t1-x1)/deltax
               elif((avg_x-x1)*(avg_x-x2)<0):
                  density[x_count]=density[x_count]+1
                  broken_count=broken_count+1
                  if(type==1):
                     density_weak[x_count]=density_weak[x_count]+1#(t1-x1)/deltax
                  elif(type==2):
                     density_strong[x_count]=density_strong[x_count]+1#(t1-x1)/deltax

               
              

         # considering all bonds- broken and non-broken
         for i in range(0, len(initial_connected_atoms)):
      ##      if(bond_broken_vs_time[int(i)]==len(time_step_list)):
            if(True): #all bonds
               
               j1=chains_init[i,1]-1#atoms 1 - one end of chain # this information is taken from the initial file
               j2=chains_init[i,2]-1#atom 2- another end of chain
               type=chains_init[i,0]
            
               t1=links_final[j1,0] #atoms 1 x-coordinates # this coordinate has been taken from the most recent file read using readLAMMPS
               t2=links_final[j2,0] #atom 2 x-coordinates

               chain_length=np.sqrt((links[j1,0]-links[j2,0])**2+(links[j1,1]-links[j2,1])**2+(links[j1,2]-links[j2,2])**2)

               if(chain_length>cnt_length_new):

                  if((t1+t2-x_L)/2.0>xlo):
                     avg_x=(t1+t2-x_L)/2.0
                  if((t1+t2+x_L)/2.0<xhi):
                     avg_x=(t1+t2+x_L)/2.0
                  #stop
               else:
                  avg_x=(t1+t2)/2.0

               

                  
               if((avg_x<xlo) or (avg_x>xhi)): # this case will never arise - because periodic bc already implememted
                  density_all_bonds[0]=density_all_bonds[0]+1
                  all_bonds_count=all_bonds_count+1
                  if(type==1):
                     density_all_bonds_weak[0]=density_all_bonds_weak[0]+1#(t1-x1)/deltax
                  elif(type==2):
                     density_all_bonds_strong[0]=density_all_bonds_strong[0]+1#(t1-x1)/deltax
               elif((avg_x-x1)*(avg_x-x2)<0):
                  density_all_bonds[x_count]=density_all_bonds[x_count]+1
                  all_bonds_count=all_bonds_count+1
                  if(type==1):
                     density_all_bonds_weak[x_count]=density_all_bonds_weak[x_count]+1#(t1-x1)/deltax
                  elif(type==2):
                     density_all_bonds_strong[x_count]=density_all_bonds_strong[x_count]+1#(t1-x1)/deltax

               
##      print('density',density[1:])
##      print('density_all_bonds',density_all_bonds[1:])
      return xrange,density,density_weak,density_strong,density_all_bonds,density_all_bonds_weak,density_all_bonds_strong, broken_count,all_bonds_count







###back to main code, outside function


high_threshold_broken_frac=0.8
low_threshold_broken_frac=0.5


##step_broken=550
##[xrange,density,density_weak,density_strong,density_all_bonds,density_all_bonds_weak,density_all_bonds_strong,broken_count,all_bonds_count]=calc_broken_density(step_broken)

obtained_final_step=False
count_1=0
count_2=0
while(obtained_final_step==False):
   
   [xrange,density,density_weak,density_strong,density_all_bonds,density_all_bonds_weak,density_all_bonds_strong,broken_count,all_bonds_count]=calc_broken_density(step_broken)
   print('inside while loop')
   frac_broken=np.divide(density[1:],density_all_bonds[1:])
   print('max_frac_broken',np.max(frac_broken))
   if(np.max(frac_broken)>high_threshold_broken_frac and count_2==0): # if count_2=1, and the code some\s to this point, it means that it is just going back and forth and will run into an infinite while loop
      step_broken=step_broken-step_delta ##step_Delta is limited by the simulation data
      count_1=count_1+1
      print('Exceeds high threshold, decreasing step_broken')
      [xrange,density,density_weak,density_strong,density_all_bonds,density_all_bonds_weak,density_all_bonds_strong,broken_count,all_bonds_count]=calc_broken_density(step_broken)

   elif(np.max(frac_broken)<low_threshold_broken_frac):
      if(count_1==1):
         step_broken=step_broken+step_delta ## this means that broken fraction first exceeded threshold, step_broken was decreased, and now it is below threshold. in this case- I will err on the higher side
         [xrange,density,density_weak,density_strong,density_all_bonds,density_all_bonds_weak,density_all_bonds_strong,broken_count,all_bonds_count]=calc_broken_density(step_broken)

         print('tried decreasing step_broken after increasing once, hence breaking... err on the higher side')
         print('obtained final step:',step_broken)
         break
      elif(count_1==0):
         step_broken=step_broken+step_delta
         count_2=count_2+1
         if(step_broken>step_broken_1):
            print('cannot go further!')
            break  # cannot do anything in this case even though broken_fraction is below threshold- because that is the resolution of the simulation data
         #step_broken cannot exceed step_broken_1 because step_broken_1 is the maximum time step upto which network exists
         print('Below low threshold, increasing step_broken')

         [xrange,density,density_weak,density_strong,density_all_bonds,density_all_bonds_weak,density_all_bonds_strong,broken_count,all_bonds_count]=calc_broken_density(step_broken)
   else:
      obtained_final_step=True # obtained_final_step. Yayy!
      nan_bool=False
      a=np.divide(density[1:],density_all_bonds[1:])
      for j in range(len(a)):
         if(np.isnan(a[j])):
            nan_bool=True
      if(nan_bool==True):
         print('nan value encountered')
         step_broken=step_broken-step_delta
         [xrange,density,density_weak,density_strong,density_all_bonds,density_all_bonds_weak,density_all_bonds_strong,broken_count,all_bonds_count]=calc_broken_density(step_broken)

      print('obtained final step:',step_broken)
      break

##print(np.divide(density[1:],density_all_bonds[1:]))
##stop
plt.figure()

plt.plot(xrange[1:],density[1:],'o-')

plt.plot(xrange[1:],density_weak[1:])

plt.plot(xrange[1:],density_strong[1:])
plt.xlabel('Tensile axis position')
plt.ylabel('Broken bonds_density')
plt.savefig('density_x_broken_new.png')

plt.figure()

plt.plot(xrange[1:],density_all_bonds[1:],'o-')

plt.plot(xrange[1:],density_all_bonds_weak[1:])

plt.plot(xrange[1:],density_all_bonds_strong[1:])
plt.xlabel('Tensile axis position')
plt.ylabel('All bonds_density')
plt.savefig('density_x_all_bonds_new.png')

plt.figure()
plt.plot(xrange[1:],np.divide(density[1:],density_all_bonds[1:]),'o-')
plt.xlabel('Tensile axis position')
plt.ylabel('Fraction broken')
plt.savefig('fraction_broken_new.png')
np.savetxt('bond_breaking_data.txt',np.transpose(np.array([xrange[1:],density[1:],density_weak[1:],density_strong[1:],density_all_bonds[1:],density_all_bonds_weak[1:],density_all_bonds_strong[1:],np.divide(density[1:],density_all_bonds[1:])])),comments='[xrange[1:],density[1:],density_weak[1:],density_strong[1:],density_all_bonds[1:],density_all_bonds_weak[1:],density_all_bonds_strong[1:],np.divide(density[1:],density_all_bonds[1:])]')
print('broken_count',broken_count)
print('all_bonds_count',all_bonds_count)
##plt.show()
