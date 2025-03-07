U0_low=20
U0_high=20
lam_max=4


epsilon=1e-5
sigma=4
U0_EV=1

N_low=12
N_high=12
N=N_low


b_low=1.0
b_high=1.0
b=b_low

nu=1.0
C=1
factor=40
##frac_weak=0.0

##L=62.96 # corresponding to dimensionless conc=3.163

del_t=0.002
e_rate=5


cR3=3 # dimless conc
n_chains=10000

conc=cR3/(N*b**2)**1.5 #(chains/nm3)
L=(n_chains/conc)**(1/3)
C_mM=conc/0.6022 # conc in mM
print('L',L)
print('C_mM', C_mM)


K_low=1.0
K_high=1.0

fit_param_low=1.0
fit_param_high=1.0

E_b_low=1200.0
E_b_high=1200.0

func=4

tol=0.01
max_itr = 100000
write_itr = 1000
wrt_step = 1
