## python file to calculate the width of the network along the y and z axes, and write data- for all replicates####

## Written by Devosmita Sen##
## August 2024 ##
## This file considers the filesystem as follows:

#current working directory
## |
##  -- Run1
## |
##  -- Run2
## ...




def readLAMMPS_after_netgen(filename, vflag,G):

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

   line6 = f1.readline()
           
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

   for c in chains:
      [lnk_1,lnk_2]=c[2:4]
      G.add_edge(lnk_1,lnk_2)
##   directory = './'+str(int(100*frac_weak))+'/'
   ##filename = './0/primary_loops'
##   file_path = os.path.join(directory, filename)
##   if not os.path.isdir(directory):
##      os.mkdir(directory)  
   loop_atoms = [] ##np.genfromtxt(filename , usecols=(1), skip_header=0)
   ##loop_atoms.tolist() 

   return xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, links, chains, atom_types, bond_types, mass, loop_atoms,G


import io
import matplotlib
##matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import numpy as np
import sys
import networkx as nx
##import ioLAMMPS_new as ioLAMMPS
from scipy.optimize import curve_fit
import math
from sklearn.cluster import KMeans

from scipy import interpolate
import Run1.param as p

frac_weak=p.frac_weak_arr[0]

def func(x, a):
    return a/(1+a*x)##np.sqrt(x)

threshold_arr=np.array([0.0,0.1,0.2,0.3,0.4,0.6,0.8,0.95])##np.array([-0.2,-0.1,-0.05,0.0,0.02,0.04,0.05,0.07,0.09,0.1,0.15,0.2,0.3])#,0.4,0.5,0.6,0.8,0.99])##[-0.4,-0.1,0.0,0.2,0.4,0.6,0.8,0.99]##[-0.4,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
##for threshold in threshold_arr:
from multiprocessing import Pool


        
def calc_width(threshold):
    ##for threshold in threshold_arr:
    
    G=nx.Graph()
    vflag=0


    run_arr=np.arange(1,11)##[1,2,3,4,5,6,7,8,9,10]
    ##threshold_arr=[-0.4,-0.2,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]##,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    ##threshold=0.05
    filename="./Run1"+"/"+str(int(frac_weak*100))+"/restart_network_"+str(0)+".txt"
    [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
                  atom_types, bond_types, mass, loop_atoms,G] = readLAMMPS_after_netgen(filename, vflag,G)
    delta_x_plot=0.3 ##0.3##
    deltax=delta_x_plot
    xlo_0=xlo
    xhi_0=xhi
    ite_arr=np.arange(0,15,1)
    density_all_y=np.zeros((len(ite_arr),len(run_arr),math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)+1))
    density_all_z=np.zeros((len(ite_arr),len(run_arr),math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)+1))
    run_cnt=-1
    for run in run_arr:
       run_cnt=run_cnt+1
        #threshold_arr=[0.0] ##-0.4,-0.2,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]##,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        

       print('Run='+str(run))
       directory="./Run"+str(run)
       x_width_old=0
       y_width_old=0
       z_width_old=0



       filename=directory+"/"+str(int(frac_weak*100))+"/restart_network_"+str(0)+".txt"
       try:
        [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
                      atom_types, bond_types, mass, loop_atoms,G] = readLAMMPS_after_netgen(filename, vflag,G)
        ##stop
        xlo_0=xlo
        xhi_0=xhi

        Ly0=yhi-ylo
        Lz0=zhi-zlo

        x_width_old=xhi-xlo
        y_width_old=yhi-ylo
        z_width_old=zhi-zlo

        x_width_old_ideal=xhi-xlo
        y_width_old_ideal=yhi-ylo
        z_width_old_ideal=zhi-zlo

        vol_init=x_width_old**3

        poisson_y_arr=[]
        poisson_z_arr=[]

        poisson_y_ideal_arr=[]
        poisson_z_ideal_arr=[]


      

        import glob

        
        vol_arr=[]


        width_y=np.zeros(len(ite_arr))
        width_z=np.zeros(len(ite_arr))


        ##y_width_arr=[38.4-4.188, 36.52-4.8, 36.1285-5.9,35.85-7.05, 34.16-7.33, 34.27-7.57, 33.71-7.8, 33.65-8.18, 33.74-8.17]
        ##y_width_arr=[33.11, 31.22, 30.02, 28.8, 27.5, 26.75, 25.9, 25.3, 24.5]
        ##y_orig_arr=y_width_arr



        ##stop
        x_width_arr=[]
        x_width_arr.append(xhi-xlo)
        y_width_arr=np.zeros(len(ite_arr))
        z_width_arr=np.zeros(len(ite_arr))

        y_width_ideal_arr=np.zeros(len(ite_arr))
        z_width_ideal_arr=np.zeros(len(ite_arr))

        mean_density_y=[]
        mean_density_z=[]

        ideal_density_y=[]
        ideal_density_z=[]


        cnt=0
        for ite in ite_arr:
          

            filename=directory+"/"+str(int(frac_weak*100))+"/restart_network_"+str(ite)+".txt"
            [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
                      atom_types, bond_types, mass, loop_atoms,G] = readLAMMPS_after_netgen(filename, vflag,G)
            #print('n_chains',n_bonds)
            ##stop
            largest_cc = max(nx.connected_components(G), key=len)
            G_conn1=G.subgraph(largest_cc)
            G_conn=G_conn1.copy()

      



            # density along y and z axis:

            #####   DENSITY ALONG Y   ############
            delta_x_plot=0.3 ##0.3##
            deltax=delta_x_plot
            density=np.zeros(math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)+1)


            x_count=-1 # we are pulling in x direction, not z
            xrange=np.zeros(len(density))#[z for z in range(zlo,zhi,deltax)]
            x2_pos=-xhi_0
            tensor=np.zeros((n_bonds,math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)))
            #print('math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)',math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3))

            for temp in range(0,math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)):#int(zlo),int(zhi),deltax):
               x_count=x_count+1
               forces=[] # stores the values of forces on all chains passign through this x plane/bin
               forces_weak=[]
               forces_strong=[]
               extensions=[]
            ##   z=z
            ##   z=xrange
               x1=x2_pos
               x2_pos=x1+delta_x_plot
               x2=x1+deltax

               xrange[x_count]=x2
               #print('x2',x2)
            ##   print('xcount',x_count)
               img_cnt=0
               chains=bonds
               links=atoms
               for i in range(0, n_bonds):
            ##      if(bond_broken_vs_time[int(i)]==len(time_step_list)):
                  if(True):
                     j1=chains[i,2]-1#atoms 1 - one end of chain
                     j2=chains[i,3]-1#atom 2- another end of chain
                     ctype=chains[i,0]

                     t1=links[j1,1] #atoms 1 y-coordinates
                     t2=links[j2,1] #atom 2 y-coordinates
        ##                             avg_x=(t1+t2)/2.0
        ####             print(t1,t2)

                     # THIS IS DOEN ASSUMING THAT THE CROSSLINKERS 1 AND 2 ARE ORDERED 
                     image=False

                     length=abs(t1-t2)


                     chain_length= length  # no PBC on y and z  ##abs(length - int(round(length/Lx))*Lx)
                     if(length!=chain_length):
                             image=True

                     number_of_bins=int(chain_length/deltax)+1


                     assigned=False
                     if(True):#chain_length>deltax):
                        if(image==False and (t1-x1)*(t2-x2)<0):#(avg_x-x1)*(avg_x-x2)<0):
                         # chain length is greater than bin length and the chain ends are on opposite sides of the bin ends(meaning that the chain crosses the bin)
                                density[x_count]=density[x_count]+1/number_of_bins
                                tensor[i,x_count]=tensor[i,x_count]+1
                                assigned=True

                        elif(image==True and (t1-x1)*(t2-x2)>0):#(avg_x-x1)*(avg_x-x2)<0):
                                img_cnt=img_cnt+1
                         # chain length is greater than bin length and the chain ends are on opposite sides of the bin ends(meaning that the chain crosses the bin)
                                density[x_count]=density[x_count]+1/number_of_bins
                                tensor[i,x_count]=tensor[i,x_count]+1
                                assigned=True

                     if(assigned==False):##chain_length<deltax):
                        if(image==False and ((t1-x1)*(t2-x1))*((t1-x2)*(t2-x2))<0):#(avg_x-x1)*(avg_x-x2)<0):
                         # chain length is greater than bin length and the chain ends are on opposite sides of the bin ends(meaning that the chain crosses the bin)
                            density[x_count]=density[x_count]+1/number_of_bins
                            tensor[i,x_count]=tensor[i,x_count]+1
                            assigned=True



            x=density##[0:-1]


            width_ideal=yhi-ylo

       
            
            xidx=np.intersect1d(np.where(xrange>=xlo_0+(xhi_0-xlo_0)*(threshold)*0.5)[0],np.where(xrange<=xhi_0-(xhi_0-xlo_0)*(threshold)*0.5)[0])


            for i in range(0,n_bonds):

                if(np.sum(tensor[i,:])!=0):
                    tensor[i,:]=tensor[i,:]/np.sum(tensor[i,:])
                
            new_x=np.zeros(np.shape(density))
            
            for cnt_t in range(0,math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)):
                new_x[cnt_t]=np.sum(tensor[:,cnt_t])
            intg_new=np.sum(new_x)
            

            x=new_x
            density_all_y[cnt,run_cnt,:]=x


            temp=x[xidx]
            mean_density=np.sum(temp[np.where(temp>0)[0]])*delta_x_plot/((xhi_0-xlo_0)*(1-threshold))
        ####    mean_density=np.mean(x[np.where(x>threshold*np.max(x)*(Ly0/width_ideal)**6)])
            intg=np.sum(x)*delta_x_plot
            #print('intg',intg)
            ##print('np.sum(x) y',np.sum(x))
            width_y[cnt]=intg/mean_density


            y_width_ideal_arr[cnt]=yhi-ylo
            ideal_density=intg/(yhi-ylo)

            ideal_density_y.append(ideal_density)

        ####    width_y[cnt]=intg/ideal_density

            #print('width[cnt]- y', width_y[cnt])
            #print('width y ideal', yhi-ylo)

            #print('mean_density-ideal_density=', mean_density-ideal_density)

            mean_density_y.append(mean_density)



            plt.figure(1)

            plt.plot(xrange,x,'o-')
        ####    plt.plot(xrange,np.ones(len(xrange))*threshold*np.max(x), label='threshold*np.max(x)')
            plt.plot(xrange,np.ones(len(xrange))*mean_density, label='mean_density')
            plt.plot(xrange,np.ones(len(xrange))*ideal_density, label='ideal_density')
            

            plt.title('y')
            plt.legend()
            
            ########## DENSITY ALONG Z  #############






            density=np.zeros(math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)+1)
        


            x_count=-1 # we are pulling in x direction, not z
            xrange=np.zeros(len(density))#[z for z in range(zlo,zhi,deltax)]
            x2_pos=-xhi_0

            tensor=np.zeros((n_bonds,math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)))

            for temp in range(0,math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)):#int(zlo),int(zhi),deltax):
               x_count=x_count+1
               forces=[] # stores the values of forces on all chains passign through this x plane/bin
               forces_weak=[]
               forces_strong=[]
               extensions=[]
            ##   z=z
            ##   z=xrange
               x1=x2_pos
               x2_pos=x1+delta_x_plot
               x2=x1+deltax

               xrange[x_count]=x2
            ##   print('xcount',x_count)
               img_cnt=0
               chains=bonds
               links=atoms
               for i in range(0, n_bonds):
            ##      if(bond_broken_vs_time[int(i)]==len(time_step_list)):
                  if(True):
                     j1=chains[i,2]-1#atoms 1 - one end of chain
                     j2=chains[i,3]-1#atom 2- another end of chain
                     ctype=chains[i,0]

                     t1=links[j1,2] #atoms 1 z-coordinates
                     t2=links[j2,2] #atom 2 z-coordinates
        ##                             avg_x=(t1+t2)/2.0

                     # THIS IS DOEN ASSUMING THAT THE CROSSLINKERS 1 AND 2 ARE ORDERED 
                     image=False

                     length=abs(t1-t2)


                     chain_length= length  # no PBC on y and z  ##abs(length - int(round(length/Lx))*Lx)
                     if(length!=chain_length):
                             image=True

                     number_of_bins=int(chain_length/deltax)+1

                     assigned=False
                     if(True):
                        if(image==False and (t1-x1)*(t2-x2)<0):#(avg_x-x1)*(avg_x-x2)<0):
                         # chains don't cross through PBC, and pass thgough bin
                                density[x_count]=density[x_count]+1/number_of_bins
                                tensor[i,x_count]=tensor[i,x_count]+1
                                assigned=True

                        elif(image==True and (t1-x1)*(t2-x2)>0):#(avg_x-x1)*(avg_x-x2)<0):
                                img_cnt=img_cnt+1
                         ## chains cross through PBC
                                density[x_count]=density[x_count]+1/number_of_bins
                                tensor[i,x_count]=tensor[i,x_count]+1
                                assigned=True

                     if(assigned==False):##chain_length<deltax):
                        if(image==False and ((t1-x1)*(t2-x1))*((t1-x2)*(t2-x2))<0):#(avg_x-x1)*(avg_x-x2)<0):
                         ## don't cross through PBC, don't pass through bin
                            density[x_count]=density[x_count]+1/number_of_bins
                            tensor[i,x_count]=tensor[i,x_count]+1
                            assigned=True


            x=density##[0:-1]
                    xidx=np.intersect1d(np.where(xrange>=xlo_0+(xhi_0-xlo_0)*(threshold)*0.5)[0],np.where(xrange<=xhi_0-(xhi_0-xlo_0)*(threshold)*0.5)[0])

           

            for i in range(0,n_bonds):
                if(np.sum(tensor[i,:])!=0):
                    tensor[i,:]=tensor[i,:]/np.sum(tensor[i,:])
               
            new_x=np.zeros(np.shape(density))
           
            for cnt_t in range(0,math.floor(int((xhi_0-xlo_0)/delta_x_plot)*3)):
                new_x[cnt_t]=np.sum(tensor[:,cnt_t])

            intg_new=np.sum(new_x)
            #print('np.sum(new_x)',np.sum(new_x))

            x=new_x
            density_all_z[cnt,run_cnt,:]=x


            temp=x[xidx]
            mean_density=np.sum(temp[np.where(temp>0)[0]])*delta_x_plot/((xhi_0-xlo_0)*(1-threshold)) ##np.mean(temp[np.where(temp>0)[0]])
        ##  mean_density=np.mean(x[np.where(x>threshold*np.max(x)*(Lz0/width_ideal)**6)])
            intg=np.sum(x)*delta_x_plot
            ##print('np.sum(x) z',np.sum(x))
            width_z[cnt]=intg/mean_density

            mean_density_z.append(mean_density)



            z_width_ideal_arr[cnt]=zhi-zlo
            ideal_density=intg/(zhi-zlo)

            ideal_density_z.append(ideal_density)

       

            width_ideal=zhi-zlo
        
        ##    max_x=0
            max_y=ylo
            max_z=zlo
            min_y=yhi
            min_z=zhi
            for lnk in G_conn.nodes():
                [x,y,z]=atoms[lnk-1,:]
                min_y=min(min_y,y)
                min_z=min(min_z,z)
                max_y=max(max_z,z)
                max_z=max(max_z,z)
        ##    print('lnk',lnk)
            y_width_new=width_y[cnt] ##max_y-min_y  #y_width_arr[cnt]##max_y-min_y
            y_width_arr[cnt]=y_width_new##max_y-min_y

            z_width_new=width_z[cnt] 
            z_width_arr[cnt]=z_width_new##max_z-min_z
            cnt=cnt+1
            ##max_z-min_z #y_width_new  #max_z-min_z

            x_width_new=xhi-xlo

            x_width_new_ideal=xhi-xlo
            y_width_new_ideal=yhi-ylo
            z_width_new_ideal=zhi-zlo

        

            vol=x_width_new*y_width_new*z_width_new

            #print('vol',vol)
            vol_arr.append(vol)
            
            ##stop

        # now threshold array =width averaged over=1-threshold

   

        plt.figure()

        ax = plt.gca()
        fontsize=15
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        plt.xticks(weight = 'bold')
        plt.yticks(weight = 'bold')

        plt.plot(ite_arr[0:],y_width_arr[0:],'o-',label='y', markersize=8, linewidth=2)

        plt.plot(ite_arr[0:],z_width_arr[0:],'o-',label='z', markersize=8, linewidth=2)

        plt.plot(ite_arr[0:],y_width_ideal_arr[0:],'o-',label='ideal', markersize=8, linewidth=2)

        ##plt.plot(ite_arr[0:],z_width_ideal_arr[0:],'o-',label='z_ideal')
        plt.xlabel('Iteration',fontsize=fontsize,fontweight='bold')
        plt.ylabel('Width',fontsize=fontsize,fontweight='bold')
        plt.legend()
        ##plt.savefig('init_relax_width')


        plt.figure()

        ax = plt.gca()
        fontsize=15
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        plt.xticks(weight = 'bold')
        plt.yticks(weight = 'bold')

        plt.plot(ite_arr[0:],mean_density_y[0:],'o-',label='y', markersize=8, linewidth=2)

        plt.plot(ite_arr[0:],mean_density_z[0:],'o-',label='z', markersize=8, linewidth=2)
        plt.plot(ite_arr[0:],ideal_density_y[0:],'o-',label='ideal y', markersize=8, linewidth=2)

        plt.plot(ite_arr[0:],ideal_density_z[0:],'o-',label='ideal z', markersize=8, linewidth=2)

        np.savetxt(directory+'/mean_density_y_tensile_'+str(threshold)+'_test.txt',np.transpose(np.array([ite_arr,mean_density_y])))
        np.savetxt(directory+'/mean_density_z_tensile_'+str(threshold)+'_test.txt',np.transpose(np.array([ite_arr,mean_density_z])))

        np.savetxt(directory+'/ideal_density_y_tensile_'+str(threshold)+'_test.txt',np.transpose(np.array([ite_arr,ideal_density_y])))
        np.savetxt(directory+'/ideal_density_z_tensile_'+str(threshold)+'_test.txt',np.transpose(np.array([ite_arr,ideal_density_z])))

        np.savetxt(directory+'/width_y_tensile_'+str(threshold)+'_test.txt',np.transpose(np.array([ite_arr,y_width_arr])))
        np.savetxt(directory+'/width_z_tensile_'+str(threshold)+'_test.txt',np.transpose(np.array([ite_arr,z_width_arr])))

        np.savetxt(directory+'/ideal_width_y_tensile_'+str(threshold)+'_test.txt',np.transpose(np.array([ite_arr,y_width_ideal_arr])))
        ##np.savetxt(directory+'/ideal_width_z_after_relax.txt',np.transpose(np.array([threshold_arr,ideal_density_z])))



        ##plt.plot(threshold_arr[0:],y_width_ideal_arr[0:],'o-',label='ideal', markersize=8, linewidth=2)

        ##plt.plot(ite_arr[0:],z_width_ideal_arr[0:],'o-',label='z_ideal')
        plt.xlabel('Iteration',fontsize=fontsize,fontweight='bold')
        plt.ylabel('Mean density',fontsize=fontsize,fontweight='bold')
        plt.legend()


        plt.figure()

        ax = plt.gca()
        fontsize=15
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        plt.xticks(weight = 'bold')
        plt.yticks(weight = 'bold')

        plt.plot(ite_arr[0:],vol_arr[0:],'o-',label='vol', markersize=8, linewidth=2)
        plt.plot(ite_arr[0:],vol_init*np.ones(len(ite_arr[0:])),'o-',label='vol', markersize=8, linewidth=2)
        plt.xlabel('Iteration',fontsize=fontsize,fontweight='bold')
        plt.ylabel('Volume',fontsize=fontsize,fontweight='bold')
        plt.legend()
        ##plt.savefig('init_relax_volume')
        plt.show()
       except:
        continue
    ite_cnt=0
    for ite in ite_arr:
        density_mean_y=np.mean(density_all_y[ite_cnt,:,:],axis=0)
        density_mean_z=np.mean(density_all_z[ite_cnt,:,:],axis=0)
        density_std_y=np.std(density_all_y[ite_cnt,:,:],axis=0)
        density_std_z=np.std(density_all_z[ite_cnt,:,:],axis=0)
        
        print(density_all_y[ite_cnt,:,:]-density_all_z[ite_cnt,:,:])
        ##print(density_all_z[ite_cnt,:,:])

        np.savetxt('density_y_z_ite_'+str(ite)+'.txt',np.transpose(np.array([xrange,density_mean_y,density_std_y,density_mean_z,density_std_z])) )
        ite_cnt=ite_cnt+1

if __name__ == '__main__':
    with Pool(len(threshold_arr)) as p:
        print(p.map(calc_width, threshold_arr))
        
##calc_width(threshold_arr)
ite_cnt=0
for ite in ite_arr:
    density_mean_y=np.mean(density_all_y[ite_cnt,:,:],axis=0)
    density_mean_z=np.mean(density_all_z[ite_cnt,:,:],axis=0)
    density_std_y=np.std(density_all_y[ite_cnt,:,:],axis=0)
    density_std_z=np.std(density_all_z[ite_cnt,:,:],axis=0)

    print(density_all_y[ite_cnt,:,:]-density_all_z[ite_cnt,:,:])
    ##print(density_all_z[ite_cnt,:,:])

    np.savetxt('density_y_z_ite_'+str(ite)+'.txt',np.transpose(np.array([xrange,density_mean_y,density_std_y,density_mean_z,density_std_z])) )
    ite_cnt=ite_cnt+1
