import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse
import copy
import PEPS_simple as PEPS

def set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D):
    return PEPS.set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position='bulk')

def simple_update(Tn,lam,expH,chi_max,Lx, Ly,inv_precision=1e-10):
    return PEPS.simple_update(Tn,lam,expH,chi_max,Lx, Ly,inv_precision=inv_precision,periodic=True)

def simple_update_second(Tn,lam,expH,expH2,chi_max,Lx, Ly,inv_precision=1e-10):
    return PEPS.simple_update_second(Tn,lam,expH,expH2,chi_max,Lx, Ly,inv_precision=inv_precision,periodic=True)

def Contract_one_site(Tn_i,lam_i,op):
    return PEPS.Contract_one_site(Tn_i,lam_i,op)

def Contract_one_site_no_op(Tn_i,lam_i):
    return PEPS.Contract_one_site_no_op(Tn_i,lam_i)

def Contract_two_site(Tn_i, Tn_j, lam_i, lam_j, op1, op2, dir='horizontal'):
    return PEPS.Contract_two_site(Tn_i, Tn_j, lam_i, lam_j, op1, op2, dir=dir)

def Contract_two_site_no_op(Tn_i,Tn_j,lam_i,lam_j,dir='horizontal'):
    return PEPS.Contract_two_site_no_op(Tn_i,Tn_j,lam_i,lam_j,dir=dir)

def Calc_mag(Tn,lam):
    return PEPS.Calc_mag(Tn,lam)

def Calc_dot(Tn,lam,Sz,Sp,Sm, Lx, Ly):
    return PEPS.Calc_dot(Tn,lam,Sz,Sp,Sm,Lx,Ly,periodic=True)

def Calc_Energy(Tn,lam,Jz,Jxy,hx,hz,D, Lx, Ly):
    return PEPS.Calc_Energy(Tn,lam,Jz,Jxy,hx,hz,D,Lx,Ly,periodic=True)

def make_expH_list(Ham, tau, Lx, Ly, m):
    ## Imaginary time evolution operator U
    N = Lx * Ly

    expH_temp = linalg.expm(-tau*Ham).reshape(m,m,m,m)
    expH = []
    for i in range(N):
        expH_local = [expH_temp, expH_temp] 
        expH.append(expH_local)
    return expH


def simple_IT_Simulation(m,Jz,Jxy,hx,hz,D,Lx,Ly,chi_max,tau_max,tau_min,tau_step,inv_precision=1e-10,second_ST=False,tensor_dtype=np.dtype(float),output_dyn=False,output_dyn_num=100,output=False):

    tau_factor = (tau_min/tau_max)**(1.0/tau_step)
    output_step = tau_step//output_dyn_num
    
    Ham = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D)
    
    ## AF initial condition
    Tn = []
    lam = []

    N = Lx * Ly
    for iy in range(Ly):
        for ix in range(Lx):
            Tn.append(np.zeros((1,1,1,1,m),dtype=tensor_dtype))
            i = ix + iy * Lx
            if ( (ix + iy) % 2 == 0):
                Tn[-1][0,0,0,0,0] = 1.0
            else:
                Tn[-1][0,0,0,0,m-1] = 1.0
            lam_local=[np.ones((1),dtype=float),np.ones((1),dtype=float),np.ones((1),dtype=float),np.ones((1),dtype=float)]
            lam.append(lam_local)

    ## Imaginary time evolution (TEBD algorithm)
    tau = tau_max
    T = 0.0
    T_list = []
    E_list = []
    mz_list = []

    
    for n in range(tau_step):
        ## make expH list
        expH = make_expH_list(Ham, tau, Lx, Ly, m)
        if (second_ST):
            expH2 = make_expH_list(Ham, 0.5 * tau, Lx, Ly, m)

        if (output_dyn and n%output_step == 0):
            
            mz = Calc_mag(Tn, lam)
            E = Calc_Energy(Tn, lam,Jz, Jxy,hx,hz, D, Lx, Ly)
            print(f"##Dyn {T} {E/N} {np.sum(mz)/N}")

            T_list.append(T)
            E_list.append(E/N)
            mz_list.append(mz)
            
        if second_ST:
            Tn,lam = simple_update_second(Tn,lam,expH,expH2,chi_max,Lx, Ly,inv_precision=inv_precision)
        else:
            Tn,lam = simple_update(Tn,lam,expH,chi_max,Lx, Ly,inv_precision=inv_precision)

        T += tau 
        tau = tau*tau_factor

    if output:
        mz = Calc_mag(Tn, lam)
        E = Calc_Energy(Tn, lam, Jz, Jxy, hx, hz, D, Lx, Ly)
        print(m, Jz, Jxy, hx, hz, D, E/N, np.sum(mz)/N)

    if output_dyn:
        return Tn,lam,np.array(T_list),np.array(E_list),np.array(mz_list)
    else:
        return Tn,lam

def simple_RT_Simulation(m,Jz,Jxy,hx,hz,D,Lx, Ly,chi_max,dt,t_step,init_Tn, init_lam, inv_precision=1e-10,second_ST=False,tensor_dtype=np.dtype(complex),output_dyn=False,output_dyn_num=100,output=False):
    output_step = t_step//output_dyn_num

    Ham = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D)

    Tn = copy.deepcopy(init_Tn)
    lam = copy.deepcopy(init_lam)
        
    ## Real time evolution (TEBD algorithm)
    T = 0.0
    T_list = []
    E_list = []
    mz_list = []
    
    N = Lx * Ly
    for n in range(t_step):
        ## real time evolution operator U
        expH = make_expH_list(Ham, 1.0j * dt, Lx, Ly, m)

        if (second_ST):
            expH2 = make_expH_list(Ham, 0.5j * dt, Lx, Ly, m)

        if (output_dyn and n%output_step == 0):
            mz = Calc_mag(Tn, lam)
            E = Calc_Energy(Tn, lam,Jz, Jxy,hx,hz, D, Lx, Ly)
            #print(f"##Dyn {T} {E} {np.sqrt(np.sum(mz**2)/N)} {mz}")
            print(f"##Dyn {T} {E/N} {np.sum(mz**2)/N}")
            T_list.append(T)
            E_list.append(E/N)
            mz_list.append(mz)
            
        if second_ST:
            Tn,lam = simple_update_second(Tn,lam,expH,expH2,chi_max,Lx, Ly,inv_precision=inv_precision)
        else:
            Tn,lam = simple_update(Tn,lam,expH,chi_max,Lx, Ly,inv_precision=inv_precision)

        T = (n + 1) *dt

    if output:
        mz = Calc_mag(Tn, lam)
        E = Calc_Energy(Tn, lam, Jz, Jxy, hx, hz, D, Lx , Ly)
        #print(m, Jz, Jxy, hx, hz, D, E, np.sqrt(np.sum(mz**2)/N))
        print(m, Jz, Jxy, hx, hz, D, E/N, np.sum(mz)/N)

    if output_dyn:
        return Tn,lam,np.array(T_list),np.array(E_list),np.array(mz_list)
    else:
        return Tn,lam

#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='simple update siumulator for 2d spin model')
    parser.add_argument('-Lx', metavar='Lx',dest='Lx', type=int, default=4,
                        help='set unit cell size Lx (default = 4)')
    parser.add_argument('-Ly', metavar='Ly',dest='Ly', type=int, default=4,
                        help='set unit cell size Ly (default = 4)')
    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=-1.0,
                        help='amplitude for SzSz interaction  (default = -1.0)')
    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=0.0,
                        help='amplitude for SxSx + SySy interaction  (default = 0.0)')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=2,
                        help='Spin size m=2S +1  (default = 2)' )
    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=1.5,
                        help='extarnal magnetix field along x (default = 1.5)')
    parser.add_argument('-hz', metavar='hz',dest='hz', type=float, default=0.0,
                        help='extarnal magnetix field along z (default = 0.0)')
    parser.add_argument('-D', metavar='D',dest='D', type=float, default=0.0,
                        help='single ion anisotropy Sz^2  (default = 0.0)')
    parser.add_argument('-chi', metavar='chi_max',dest='chi_max', type=int, default=6,
                        help='maximum bond dimension at truncation  (default = 6)')
    parser.add_argument('-tau_max', metavar='tau_max',dest='tau_max', type=float, default=0.1,
                        help='start imaginary time step  (default = 0.1)')
    parser.add_argument('-tau_min', metavar='tau_min',dest='tau_min', type=float, default=0.001,
                        help='final imaginary time step  (default = 0.001)')
    parser.add_argument('-tau_step', metavar='tau_step',dest='tau_step', type=int, default=2000,
                        help='ITE steps. tau decreses from tau_max to tau_min gradually  (default = 2000)')
    parser.add_argument('-inv_precision', metavar='inv_precision',dest='inv_precision', type=float, default=1e-10,
                        help='smaller singular values than inv_precision is neglected at iTEBD update  (default = 1e-10)')
    parser.add_argument('--use_complex', action='store_const', const=True,
                        default=False, help='Use complex tensors  (default = False)')
    parser.add_argument('--second_ST', action='store_const', const=True,
                        default=False, help='Use second order Suzuki Trotter decomposition  (default = False)')
    parser.add_argument('--output_dyn', action='store_const', const=True,
                        default=False, help='Output energies along ITE  (default = False)')
    parser.add_argument('-output_dyn_num', metavar='output_dyn_num',dest='output_dyn_num', type=int, default=100,
                        help='number of data points at dynamics output  (default = 100)')
    return parser.parse_args()
    

if __name__ == "__main__":
    ## read params from command line
    args = parse_args()

    if args.use_complex:
        tensor_dtype = np.dtype(complex)
    else:
        tensor_dtype = np.dtype(float)

    if args.output_dyn:
        Tn, lam, T_list, E_list, mz_list = simple_IT_Simulation(args.m,args.Jz,args.Jxy,args.hx,args.hz,args.D,args.Lx, args.Ly,args.chi_max,args.tau_max,args.tau_min,args.tau_step,args.inv_precision,args.second_ST,tensor_dtype,args.output_dyn,args.output_dyn_num,output=True)
    else:
        Tn, lam = simple_IT_Simulation(args.m,args.Jz,args.Jxy,args.hx,args.hz,args.D,args.Lx, args.Ly,args.chi_max,args.tau_max,args.tau_min,args.tau_step,args.inv_precision,args.second_ST,tensor_dtype,args.output_dyn,args.output_dyn_num,output=True)

