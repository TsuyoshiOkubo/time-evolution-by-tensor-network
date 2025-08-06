import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse
import copy

def set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position=0):
    ## m = 2S+1    
    Sp = np.zeros((m,m))
    for i in range(1,m):
        Sp[i-1,i] = np.sqrt(i * (m - i))
    
    Sm = np.zeros((m,m))
    for i in range(0,m-1):
        Sm[i+1,i] = np.sqrt((i + 1.0) * (m - 1.0 - i))

    Sz = np.zeros((m,m))
    for i in range(m):
        Sz[i,i] = 0.5 * (m - 1.0) - i
    
    Sx = 0.5 * (Sp + Sm)
    Sz2 = np.dot(Sz,Sz)

    Id = np.identity(m)

    if position == 0: ##center
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - 0.5 * hx * (np.kron(Id,Sx) + np.kron(Sx,Id)) - 0.5 * hz * (np.kron(Id,Sz) + np.kron(Sz,Id))+ 0.5 * D * (np.kron(Id,Sz2) + np.kron(Sz2,Id))
    elif position < 0: ## left boundary
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - hx * (0.5 * np.kron(Id,Sx) + np.kron(Sx,Id)) - hz * (0.5 * np.kron(Id,Sz) + np.kron(Sz,Id))+ D * (0.5 * np.kron(Id,Sz2) + np.kron(Sz2,Id))
    else: ## right boundary
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - hx * (np.kron(Id,Sx) + 0.5 * np.kron(Sx,Id)) - hz * (np.kron(Id,Sz) + 0.5 * np.kron(Sz,Id))+ D * (np.kron(Id,Sz2) + 0.5 * np.kron(Sz2,Id))

def mult_left(v,lam_i,Tn_i):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i,(0,1)),Tn_i.conj(),([0,1],[1,0]))

def mult_right(v,lam_i,Tn_i):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i,(0,2)),Tn_i.conj(),([0,1],[2,0]))

def mult_left_op(v,lam_i,Tn_i,op):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i,(0,1)),op,(1,1)),Tn_i.conj(),([0,2],[1,0]))

def TEBD_update(Tn,lam,expH,chi_max,inv_precision=1e-10):
    N = len(Tn)
    m = expH[0].shape[0]

    lam_inv =[]

    for i in range(N+1):
        lam_inv.append(1.0/lam[i])

    for eo in range(2):
        for i in range(eo,N-1,2):
            ## apply expH
            chi_l = Tn[i].shape[1]
            chi_r = Tn[i+1].shape[2]

            Theta = np.tensordot(
                np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                np.diag(lam[i]),Tn[i],(1,1))
                            ,np.diag(lam[i+1]),(2,0))
                        ,Tn[i+1],(2,1))
                    ,np.diag(lam[i+2]),(3,0))
                ,expH[i],([1,2],[2,3])
            ).transpose(0,2,1,3).reshape(chi_l*m,chi_r*m)
            ## SVD
            U,s,VT = linalg.svd(Theta,full_matrices=False)

            ## Truncation
            ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
            chi = np.min([np.sum(s > inv_precision),chi_max])
            lam[i+1] = s[:chi]/np.sqrt(np.sum(s[:chi]**2))

            Tn[i] = np.tensordot(np.diag(lam_inv[i]),U[:,:chi].reshape(chi_l,m,chi),(1,0)).transpose(1,0,2)
            Tn[i+1] = np.tensordot(VT[:chi,:].reshape(chi,chi_r,m),np.diag(lam_inv[i+2]),(1,0)).transpose(1,0,2)

            lam_inv[i+1] = 1.0/lam[i+1]

    return Tn,lam

def TEBD_update_second(Tn,lam,expH,expH2,chi_max,inv_precision=1e-10):
    N = len(Tn)
    m = expH[0].shape[0]

    lam_inv =[]

    expH_eo=[expH2,expH]
    
    for i in range(N+1):
        lam_inv.append(1.0/lam[i])

    for eoe in range(3):
        if eoe == 1:
            eo = 1
        else:
            eo = 0    
        for i in range(eo,N-1,2):
            ## apply expH
            chi_l = Tn[i].shape[1]
            chi_r = Tn[i+1].shape[2]

            Theta = np.tensordot(
                np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                np.diag(lam[i]),Tn[i],(1,1))
                            ,np.diag(lam[i+1]),(2,0))
                        ,Tn[i+1],(2,1))
                    ,np.diag(lam[i+2]),(3,0))
                ,expH_eo[eo][i],([1,2],[2,3])
            ).transpose(0,2,1,3).reshape(m*chi_l,m*chi_r)
            ## SVD
            U,s,VT = linalg.svd(Theta,full_matrices=False)

            ## Truncation
            ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
            chi = np.min([np.sum(s > inv_precision),chi_max])
            lam[i+1] = s[:chi]/np.sqrt(np.sum(s[:chi]**2))

            Tn[i] = np.tensordot(np.diag(lam_inv[i]),U[:,:chi].reshape(chi_l,m,chi),(1,0)).transpose(1,0,2)
            Tn[i+1] = np.tensordot(VT[:chi,:].reshape(chi,chi_r,m),np.diag(lam_inv[i+2]),(1,0)).transpose(1,0,2)

            lam_inv[i+1] = 1.0/lam[i+1]

    return Tn, lam

def Calc_Environment(Tn,lam,canonical=False):
    ## Calculate left and right contraction exactly

    N = len(Tn)
    Env_left = []
    Env_right = []
    Env_right_temp = []

    if canonical:
        ## assume MPS is in canonical form
        for i in range(N):
            Env_left.append(np.identity((lam[i].shape[0])))
            Env_right.append(np.dot(np.dot(np.diag(lam[i+1]),np.identity((lam[i+1].shape[0]))),np.diag(lam[i+1])))
    else:
        left_env = np.identity(1).reshape(1,1)
        right_env = np.identity(1).reshape(1,1)
        Env_left.append(left_env)
        Env_right_temp.append(np.dot(np.dot(np.diag(lam[N]),right_env),np.diag(lam[N])))
        for i in range(1,N):

            left_env = mult_left(left_env,lam[i-1],Tn[i-1])
            right_env = mult_right(right_env,lam[N-i+1],Tn[N-i])

            Env_left.append(left_env)
            Env_right_temp.append(np.dot(np.dot(np.diag(lam[N-i]),right_env),np.diag(lam[N-i])))

        for i in range(N):
            Env_right.append(Env_right_temp[N-i-1])

    return Env_left,Env_right


def Contract_one_site(El,Er,T_i,lam_i,op):
    return np.tensordot(mult_left_op(El,lam_i,T_i,op),Er,([0,1],[0,1]))
def Contract_one_site_no_op(El,Er,T_i,lam_i):
    return np.tensordot(mult_left(El,lam_i,T_i),Er,([0,1],[0,1]))

def Contract_two_site(El,Er,T1,T2,lam1,lam2,op1,op2):
    return np.tensordot(mult_left_op(mult_left_op(El,lam1,T1,op1),lam2,T2,op2),Er,([0,1],[0,1]))
def Contract_two_site_no_op(El,Er,T1,T2,lam1,lam2):
    return np.tensordot(mult_left(mult_left(El,lam1,T1),lam2,T2),Er,([0,1],[0,1]))

def Contract_correlation(Env_left,Env_right,Tn,lam,op1,op2,max_distance,step=1):
    ## Output sequence of correlation <op1(0) op2(r)> for r <= max_distance.
    ## r is increased by step

    N = len(Tn)
    Correlation=[]
    El = Env_left[0]
    Er = Env_right[0]
    
    El_op = mult_left_op(El,lam[0],Tn[0],op1)
    El_identity = mult_left(El,lam[0],Tn[0])
    for j in range(1,step):
        El_op = mult_left(El_op,lam[j],Tn[j])
        El_identity = mult_left(El_identity,lam[j],Tn[j])
    
    for r in range(1,max_distance+1):
        El_op2 = mult_left_op(El_op,lam[step*r],Tn[step*r],op2)
        El_identity = mult_left(El_identity,lam[step*r],Tn[step*r])
        
        Correlation.append(np.real(np.tensordot(El_op2,Env_right[step*r],([0,1],[0,1]))/np.tensordot(El_identity,Env_right[step*r],([0,1],[0,1]))))
        if r < max_distance:
            El_op = mult_left(El_op,lam[step*r],Tn[step*r])
            for j in range(1,step):
                El_op = mult_left(El_op,lam[step*r + j],Tn[step*r + j])
                El_identity = mult_left(El_identity,lam[step*r + j],Tn[step*r + j])
    return Correlation

def Calc_mag(Env_left,Env_right,Tn,lam):

    N = len(Tn)
    m = Tn[0].shape[0]
    Sz = np.zeros((m,m))
    for i in range(m):
        Sz[i,i] = 0.5 * (m - 1.0) - i    
    
    mz = np.zeros(N)
    for i in range(N):
        mz[i]=np.real(Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sz)/Contract_one_site_no_op(Env_left[i],Env_right[i],Tn[i],lam[i]))
        
    return mz

def Calc_dot(Env_left,Env_right,Tn,lam,Sz,Sp,Sm):
    N = len(Tn)
    zz = np.zeros(N-1)
    pm = np.zeros(N-1)
    mp = np.zeros(N-1)
    for i in range(N-1):
        norm = Contract_two_site_no_op(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1])
        zz[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sz,Sz)/norm)
        pm[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sp,Sm)/norm)
        mp[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sm,Sp)/norm)

    return zz,pm,mp

def Calc_Energy(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,hz,D):
    N = len(Tn)
    m = Tn[0].shape[0]
    
    Sp = np.zeros((m,m))
    for i in range(1,m):
        Sp[i-1,i] = np.sqrt(i * (m - i))
    
    Sm = np.zeros((m,m))
    for i in range(0,m-1):
        Sm[i+1,i] = np.sqrt((i + 1.0) * (m - 1.0 - i))

    Sz = np.zeros((m,m))
    for i in range(m):
        Sz[i,i] = 0.5 * (m - 1.0) - i
    
    Sx = 0.5 * (Sp + Sm)
    Sz2 = np.dot(Sz,Sz)

    Id = np.identity(m)

    zz = np.zeros(N-1)
    pm = np.zeros(N-1)
    mp = np.zeros(N-1)

    mx = np.zeros(N)
    mz = np.zeros(N)
    z2 = np.zeros(N)
    
    for i in range(N):
        norm = Contract_one_site_no_op(Env_left[i],Env_right[i],Tn[i],lam[i])
        mx[i] = np.real(Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sx)/norm)
        mz[i] = np.real(Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sz)/norm)
        z2[i] = np.real(Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sz2)/norm)
    
    for i in range(N-1):
        norm = Contract_two_site_no_op(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1])
        zz[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sz,Sz)/norm)
        pm[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sp,Sm)/norm)
        mp[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sm,Sp)/norm)
    
    E = (Jz * np.sum(zz) + 0.5 * Jxy * (np.sum(pm) + np.sum(mp)) -hx * np.sum(mx) -hz * np.sum(mz) + D * np.sum(z2)) #/ (N-1)
    return E

def TEBD_IT_Simulation(m,Jz,Jxy,hx,hz,D,N,chi_max,tau_max,tau_min,tau_step,inv_precision=1e-10,second_ST=False,tensor_dtype=np.dtype(float),output_dyn=False,output_dyn_num=100,output=False):
    tau_factor = (tau_min/tau_max)**(1.0/tau_step)
    output_step = tau_step//output_dyn_num

    
    Ham = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D)
    Ham_l = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position=-1)
    Ham_r = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position=1)
    
    ## Trivial Canonical form initial condition
    Tn = []
    lam = []
    for i in range(N):
        Tn.append(np.zeros((m,1,1),dtype=tensor_dtype))
        if ( i % 2 == 0):
            Tn[i][0,0,0] = 1.0
        else:
            Tn[i][m-1,0,0] = 1.0
        lam.append(np.ones((1),dtype=float))

    lam.append(np.ones((1),dtype=float))
        
    ## Imaginary time evolution (TEBD algorithm)
    tau = tau_max
    T = 0.0
    T_list = []
    E_list = []
    mz_list = []
    S_list = []
    
    for n in range(tau_step):
        ## Imaginary time evolution operator U
        expH_l = linalg.expm(-tau*Ham_l).reshape(m,m,m,m)
        expH_c = linalg.expm(-tau*Ham).reshape(m,m,m,m)
        expH_r = linalg.expm(-tau*Ham_r).reshape(m,m,m,m)

        expH = [expH_l]
        for i in range(1,N-2):
            expH.append(expH_c)
        expH.append(expH_r)

        if (second_ST):
            expH2_l = linalg.expm(-0.5 * tau*Ham_l).reshape(m,m,m,m)
            expH2_c = linalg.expm(-0.5 * tau*Ham).reshape(m,m,m,m)
            expH2_r = linalg.expm(-0.5 * tau*Ham_r).reshape(m,m,m,m)

            expH2 = [expH2_l]
            for i in range(1,N-2):
                expH2.append(expH2_c)
            expH2.append(expH2_r)
        
        if (output_dyn and n%output_step == 0):
            Env_left,Env_right = Calc_Environment(Tn,lam)
            
            mz = Calc_mag(Env_left,Env_right,Tn,lam)
            E = Calc_Energy(Env_left,Env_right,Tn,lam,Jz, Jxy,hx,hz, D)
            prob = lam[N//2]**2
            S = -np.sum(prob * np.log(prob)) # Entropy
            #print(f"##Dyn {T} {E} {np.sqrt(np.sum(mz**2)/N)} {mz}")
            print(f"##Dyn {T} {E} {np.sum(mz)/N} {S}")

            T_list.append(T)
            E_list.append(E)
            mz_list.append(mz)
            S_list.append(S)
            
        if second_ST:
            Tn,lam = TEBD_update_second(Tn,lam,expH,expH2,chi_max,inv_precision=inv_precision)
        else:
            Tn,lam = TEBD_update(Tn,lam,expH,chi_max,inv_precision=inv_precision)

        T += tau 
        tau = tau*tau_factor

    if output:
        Env_left,Env_right = Calc_Environment(Tn,lam)        
        mz = Calc_mag(Env_left,Env_right,Tn,lam)    
        E = Calc_Energy(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,hz,D)
        prob = lam[N//2]**2
        S = -np.sum(prob * np.log(prob)) # Entropy
        #print(m, Jz, Jxy, hx, hz, D, E, np.sqrt(np.sum(mz**2)/N))
        print(m, Jz, Jxy, hx, hz, D, E, np.sum(mz)/N, S)

    if output_dyn:
        return Tn,lam,np.array(T_list),np.array(E_list),np.array(mz_list), np.array(S_list)
    else:
        return Tn,lam

def TEBD_RT_Simulation(m,Jz,Jxy,hx,hz,D,N,chi_max,dt,t_step,init_Tn, init_lam, inv_precision=1e-10,second_ST=False,tensor_dtype=np.dtype(complex),output_dyn=False,output_dyn_num=100,output=False):
    output_step = t_step//output_dyn_num

    
    Ham = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D)
    Ham_l = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position=-1)
    Ham_r = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position=1)
    
    ## initial condition
    Tn = copy.deepcopy(init_Tn)
    lam = copy.deepcopy(init_lam)
        
    ## Real time evolution (TEBD algorithm)
    T = 0.0
    T_list = []
    E_list = []
    mz_list = []
    S_list = []

    
    ## real time evolution operator U
    expH_l = linalg.expm(-1.0j* dt * Ham_l).reshape(m,m,m,m)
    expH_c = linalg.expm(-1.0j* dt * Ham).reshape(m,m,m,m)
    expH_r = linalg.expm(-1.0j* dt * Ham_r).reshape(m,m,m,m)

    expH = [expH_l]
    for i in range(1,N-2):
        expH.append(expH_c)
    expH.append(expH_r)

    if (second_ST):
        expH2_l = linalg.expm(-0.5j* dt * Ham_l).reshape(m,m,m,m)
        expH2_c = linalg.expm(-0.5j* dt * Ham).reshape(m,m,m,m)
        expH2_r = linalg.expm(-0.5j* dt * Ham_r).reshape(m,m,m,m)

        expH2 = [expH2_l]
        for i in range(1,N-2):
            expH2.append(expH2_c)
        expH2.append(expH2_r)
    for n in range(t_step):
        
        if (output_dyn and n%output_step == 0):
            Env_left,Env_right = Calc_Environment(Tn,lam)
            
            mz = Calc_mag(Env_left,Env_right,Tn,lam)
            E = Calc_Energy(Env_left,Env_right,Tn,lam,Jz, Jxy,hx,hz,D)
            prob = lam[N//2]**2
            S = -np.sum(prob * np.log(prob)) # Entropy
            #print(f"##Dyn {T} {E} {np.sqrt(np.sum(mz**2)/N)} {mz}")
            print(f"##Dyn {T} {E} {np.sum(mz)/N} {S}")

            T_list.append(T)
            E_list.append(E)
            mz_list.append(mz)
            S_list.append(S)
            
        if second_ST:
            Tn,lam = TEBD_update_second(Tn,lam,expH,expH2,chi_max,inv_precision=inv_precision)
        else:
            Tn,lam = TEBD_update(Tn,lam,expH,chi_max,inv_precision=inv_precision)

        T = (n + 1) *dt

    if output:
        Env_left,Env_right = Calc_Environment(Tn,lam)        
        mz = Calc_mag(Env_left,Env_right,Tn,lam)    
        E = Calc_Energy(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,hz,D)
        prob = lam[N//2]**2
        S = -np.sum(prob * np.log(prob)) # Entropy
        #print(m, Jz, Jxy, hx, hz, D, E, np.sqrt(np.sum(mz**2)/N))
        print(m, Jz, Jxy, hx, hz, D, E, np.sum(mz)/N, S)

    if output_dyn:
        return Tn,lam,np.array(T_list),np.array(E_list),np.array(mz_list),np.array(S_list)
    else:
        return Tn,lam

## for 2d
def set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position='bulk'):
    ## m = 2S+1    
    Sp = np.zeros((m,m))
    for i in range(1,m):
        Sp[i-1,i] = np.sqrt(i * (m - i))
    
    Sm = np.zeros((m,m))
    for i in range(0,m-1):
        Sm[i+1,i] = np.sqrt((i + 1.0) * (m - 1.0 - i))

    Sz = np.zeros((m,m))
    for i in range(m):
        Sz[i,i] = 0.5 * (m - 1.0) - i
    
    Sx = 0.5 * (Sp + Sm)
    Sz2 = np.dot(Sz,Sz)

    Id = np.identity(m)

    
    if position == 'edge_ij':
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - hx / 3.0 * (np.kron(Id,Sx) + np.kron(Sx,Id)) - hz / 3.0 * (np.kron(Id,Sz) + np.kron(Sz,Id))+ D / 3.0 * (np.kron(Id,Sz2) + np.kron(Sz2,Id))
    elif position == 'edge_i':
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - hx * (0.25 * np.kron(Id,Sx) + np.kron(Sx,Id) / 3.0) - hz * (0.25 * np.kron(Id,Sz) + np.kron(Sz,Id) / 3.0)+ D * (0.25 * np.kron(Id,Sz2) + np.kron(Sz2,Id)/3.0)
    elif position == 'edge_j':
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - hx * (np.kron(Id,Sx) /3.0 + 0.25 * np.kron(Sx,Id)) - hz * (np.kron(Id,Sz) / 3.0 + 0.25 * np.kron(Sz,Id)) + D * (np.kron(Id,Sz2) / 3.0 + 0.25 * np.kron(Sz2,Id))
    elif position == 'vertex_ij': 
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - 0.5 * hx * (np.kron(Id,Sx) + np.kron(Sx,Id)) - 0.5 * hz * (np.kron(Id,Sz) + np.kron(Sz,Id))+ 0.5 * D * (np.kron(Id,Sz2) + np.kron(Sz2,Id))
    elif position == 'vertex_i': 
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - hx * (np.kron(Id,Sx) / 3.0 + 0.5 * np.kron(Sx,Id)) - hz * (np.kron(Id,Sz) / 3.0 + 0.5 * np.kron(Sz,Id))+ D * (np.kron(Id,Sz2) / 3.0 + 0.5 * np.kron(Sz2,Id))
    elif position == 'vertex_j': 
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - hx * (0.5 * np.kron(Id,Sx) + np.kron(Sx,Id) / 3.0) - hz * (0.5 * np.kron(Id,Sz) + np.kron(Sz,Id) / 3.0)+ D * (0.5 * np.kron(Id,Sz2) + np.kron(Sz2,Id) / 3.0)
    elif position == 'bulk':
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - 0.25 * hx * (np.kron(Id,Sx) + np.kron(Sx,Id)) - 0.25 * hz * (np.kron(Id,Sz) + np.kron(Sz,Id))+ 0.25 * D * (np.kron(Id,Sz2) + np.kron(Sz2,Id))
    else:
        raise ValueError("position must be 'edge_i','edge_j','vertex_i','vertex_j','vertex_ij', or 'bulk'.")

def update_Tns(Tn,lam,expH,left,right,chi_max,inv_precision=1e-10):
    
    m = expH.shape[0]
    for i in range(left,right+1):
        Tn[i] *= lam[i][None,:,None]
    Tn[right] *= lam[right+1][None,None,:]

    lam_inv_l = 1.0 / lam[left]
    lam_inv_r = 1.0 / lam[right+1]

    temp_tensor = np.tensordot(Tn[left],Tn[left+1],([2],[1])).reshape(m*Tn[left].shape[1]*m,-1)
    for i in range(left+2, right+1):
        temp_tensor = np.tensordot(temp_tensor,Tn[i],([1],[1])).reshape(-1, Tn[i].shape[2])
    
    Theta = np.tensordot(temp_tensor.reshape(m,-1,m,Tn[right].shape[2]),expH,([0,2],[2,3])).transpose(2,0,3,1).reshape(m*Tn[left].shape[1], -1)

    ## successive SVD
    U,s,VT = linalg.svd(Theta,full_matrices=False)
    chi = np.min([np.sum(s > inv_precision),chi_max])
    lam[left+1] = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
    lam_inv = 1.0 / lam[left+1]
    s[:chi] = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
    Tn[left] = U[:,:chi].reshape(Tn[left].shape[0],Tn[left].shape[1],chi) * lam_inv_l[None,:,None]

    for i in range(left+1, right):
        Theta = (VT[:chi,:] * s[:chi,None]).reshape(chi*m,-1)
        U,s,VT = linalg.svd(Theta,full_matrices=False)
        chi = np.min([np.sum(s > inv_precision),chi_max])
        lam[i+1] = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
        s[:chi] = s[:chi]/np.sqrt(np.sum(s[:chi]**2))

        Tn[i] = U[:,:chi].reshape(-1, Tn[i].shape[0],chi).transpose(1,0,2) * lam_inv[None,:,None]
        lam_inv = 1.0 / lam[i+1]
    Tn[right] = (VT[:chi,].reshape(chi,m,Tn[right].shape[2]) * lam_inv_r[None,None,:]).transpose(1,0,2)

    Tn_list = []
    lam_list = []
    for i in range(left, right):
        Tn_list.append(Tn[i])
        lam_list.append(lam[i+1])
    Tn_list.append(Tn[right])

    return Tn_list, lam_list


def TEBD_update_2d(Tn,lam,expH,chi_max,Lx, Ly, inv_precision=1e-10):
    N = Lx * Ly
    m = expH[0][0].shape[0]

    lam_inv =[]

    for i in range(N+1):
        lam_inv.append(1.0/lam[i])

    ## horizontal update
    for eo in range(2):
        for ix in range(Lx-1):
            for iy in range(Ly):
                if (ix + iy) % 2 == eo:
                    if ix % 2 == 0:
                        i = iy + ix * Ly
                        j = i + 2 * (Ly - 1 - iy) + 1

                    else:
                        i = ix * Ly + (Ly - 1 - iy)
                        j = i + 2 * iy + 1
                    ## j > i
                    
                    Tn_list, lam_list = update_Tns(Tn,lam, expH[i][0], i, j,chi_max, inv_precision)
                    for n in range(len(Tn_list)-1):
                        Tn[i+n] = Tn_list[n]
                        lam[i+n + 1] = lam_list[n]
                        lam_inv[i+n+1] = 1.0/lam[i+n+1]
                    Tn[j] = Tn_list[-1]



    ## virtical update
    for eo in range(2):
        for ix in range(Lx):
            for iy in range(Ly-1):
                if (ix + iy) % 2 == eo:
                    if ix % 2 == 0:
                        i = iy + ix * Ly
                        j = (iy + 1) + ix * Ly
                        ## j = i + 1
                        Tnl = Tn[i] * lam[i][None,:,None]
                        Tnl *= lam[j][None,None,:]
                        Tnr = Tn[j] * lam[j+1][None,None,:]

                        Theta = np.tensordot(np.tensordot(Tnl,Tnr,([2],[1])),expH[i][1],([0,2],[2,3])).transpose(2,0,3,1).reshape(m*Tn[i].shape[1], m*Tn[j].shape[2])
                        ## SVD
                        U,s,VT = linalg.svd(Theta,full_matrices=False)

                        ## Truncation
                        ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
                        chi = np.min([np.sum(s > inv_precision),chi_max])
                        lam_new = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
                        lam[j] = lam_new
                        lam_inv[j] = 1.0/lam_new

                        Tn[i] = U[:,:chi].reshape(Tn[i].shape[0],Tn[i].shape[1],chi)*lam_inv[i][None,:,None]
                        Tn[j] = (VT[:chi,:].reshape(chi,Tn[j].shape[0],Tn[j].shape[2])*lam_inv[j+1][None,None,:]).transpose(1,0,2)

                    else:
                        i = ix * Ly + (Ly - 1 - iy)
                        j = ix * Ly + (Ly - 2 - iy)

                        ## j = i - 1
                        Tnl = Tn[j] * lam[j][None,:,None]
                        Tnl *= lam[i][None,None,:]
                        Tnr = Tn[i] * lam[i+1][None,None,:]

                        Theta = np.tensordot(np.tensordot(Tnl,Tnr,([2],[1])),expH[i][1],([0,2],[3,2])).transpose(3,0,2,1).reshape(m*Tn[j].shape[1], m*Tn[i].shape[2])
                        ## SVD
                        U,s,VT = linalg.svd(Theta,full_matrices=False)

                        ## Truncation
                        ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
                        chi = np.min([np.sum(s > inv_precision),chi_max])
                        lam_new = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
                        lam[i] = lam_new
                        lam_inv[i] = 1.0/lam_new

                        Tn[j] = U[:,:chi].reshape(Tn[j].shape[0],Tn[j].shape[1],chi)*lam_inv[j][None,:,None]
                        Tn[i] = (VT[:chi,:].reshape(chi,Tn[i].shape[0],Tn[i].shape[2])*lam_inv[i+1][None,None,:]).transpose(1,0,2)



    return Tn,lam

def TEBD_update_2d_second(Tn,lam,expH,expH2,chi_max,Lx, Ly, inv_precision=1e-10):
    N = Lx * Ly
    m = expH[0][0].shape[0]

    lam_inv =[]

    for i in range(N+1):
        lam_inv.append(1.0/lam[i])

    ## horizontal update
    for eo in range(2):
        for ix in range(Lx-1):
            for iy in range(Ly):
                if (ix + iy) % 2 == eo:
                    if ix % 2 == 0:
                        i = iy + ix * Ly
                        j = i + 2 * (Ly - 1 - iy) + 1

                    else:
                        i = ix * Ly + (Ly - 1 - iy)
                        j = i + 2 * iy + 1
                    ## j > i
                    Tn_list, lam_list = update_Tns(Tn,lam, expH2[i][0], i, j,chi_max, inv_precision)
                    for n in range(len(Tn_list)-1):
                        Tn[i+n] = Tn_list[n]
                        lam[i+n + 1] = lam_list[n]
                        lam_inv[i+n+1] = 1.0/lam[i+n+1]
                    Tn[j] = Tn_list[-1]
                    
    ## virtical update
    for eoe in range(3):
        eo = eoe % 2
        for ix in range(Lx):
            for iy in range(Ly-1):
                if (ix + iy) % 2 == eo:
                    if ix % 2 == 0:
                        i = iy + ix * Ly
                        j = (iy + 1) + ix * Ly
                        ## j = i + 1
                        Tnl = Tn[i] * lam[i][None,:,None]
                        Tnl *= lam[j][None,None,:]
                        Tnr = Tn[j] * lam[j+1][None,None,:]

                        if eo == 0:
                            # 0.5 dt
                            Theta = np.tensordot(np.tensordot(Tnl,Tnr,([2],[1])),expH2[i][1],([0,2],[2,3])).transpose(2,0,3,1).reshape(m*Tn[i].shape[1], m*Tn[j].shape[2])
                        else:
                            # dt
                            Theta = np.tensordot(np.tensordot(Tnl,Tnr,([2],[1])),expH[i][1],([0,2],[2,3])).transpose(2,0,3,1).reshape(m*Tn[i].shape[1], m*Tn[j].shape[2])
                        ## SVD
                        U,s,VT = linalg.svd(Theta,full_matrices=False)

                        ## Truncation
                        ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
                        chi = np.min([np.sum(s > inv_precision),chi_max])
                        lam_new = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
                        lam[j] = lam_new
                        lam_inv[j] = 1.0/lam_new

                        Tn[i] = U[:,:chi].reshape(Tn[i].shape[0],Tn[i].shape[1],chi)*lam_inv[i][None,:,None]
                        Tn[j] = (VT[:chi,:].reshape(chi,Tn[j].shape[0],Tn[j].shape[2])*lam_inv[j+1][None,None,:]).transpose(1,0,2)
                    else:
                        i = ix * Ly + (Ly - 1 - iy)
                        j = ix * Ly + (Ly - 2 - iy)

                        ## j = i - 1
                        Tnl = Tn[j] * lam[j][None,:,None]
                        Tnl *= lam[i][None,None,:]
                        Tnr = Tn[i] * lam[i+1][None,None,:]

                        if eo == 0:
                            # 0.5 dt
                            Theta = np.tensordot(np.tensordot(Tnl,Tnr,([2],[1])),expH2[i][1],([0,2],[3,2])).transpose(3,0,2,1).reshape(m*Tn[j].shape[1], m*Tn[i].shape[2])
                        else:
                            # dt
                            Theta = np.tensordot(np.tensordot(Tnl,Tnr,([2],[1])),expH[i][1],([0,2],[3,2])).transpose(3,0,2,1).reshape(m*Tn[j].shape[1], m*Tn[i].shape[2])
                        ## SVD
                        U,s,VT = linalg.svd(Theta,full_matrices=False)

                        ## Truncation
                        ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
                        chi = np.min([np.sum(s > inv_precision),chi_max])
                        lam_new = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
                        lam[i] = lam_new
                        lam_inv[i] = 1.0/lam_new

                        Tn[j] = U[:,:chi].reshape(Tn[j].shape[0],Tn[j].shape[1],chi)*lam_inv[j][None,:,None]
                        Tn[i] = (VT[:chi,:].reshape(chi,Tn[i].shape[0],Tn[i].shape[2])*lam_inv[i+1][None,None,:]).transpose(1,0,2)

    ## horizontal update
    for eo in (1,0):
        for ix in range(Lx-1):
            for iy in range(Ly):
                if (ix + iy) % 2 == eo:
                    if ix % 2 == 0:
                        i = iy + ix * Ly
                        j = i + 2 * (Ly - 1 - iy) + 1

                    else:
                        i = ix * Ly + (Ly - 1 - iy)
                        j = i + 2 * iy + 1
                    ## j > i
                    Tn_list, lam_list = update_Tns(Tn,lam, expH2[i][0], i, j,chi_max, inv_precision)
                    for n in range(len(Tn_list)-1):
                        Tn[i+n] = Tn_list[n]
                        lam[i+n + 1] = lam_list[n]
                        lam_inv[i+n+1] = 1.0/lam[i+n+1]
                    Tn[j] = Tn_list[-1]

    return Tn,lam
def make_expH_list(Ham_b, Ham_eij, Ham_ei, Ham_ej, Ham_vij, Ham_vi, Ham_vj, tau, Lx, Ly, m):
    ## Imaginary time evolution operator U
    N = Lx * Ly

    expH_b = linalg.expm(-tau*Ham_b).reshape(m,m,m,m)
    expH_ei = linalg.expm(-tau*Ham_ei).reshape(m,m,m,m)
    expH_ej = linalg.expm(-tau*Ham_ej).reshape(m,m,m,m)
    expH_eij = linalg.expm(-tau*Ham_eij).reshape(m,m,m,m)
    expH_vij = linalg.expm(-tau*Ham_vij).reshape(m,m,m,m)
    expH_vi = linalg.expm(-tau*Ham_vi).reshape(m,m,m,m)
    expH_vj = linalg.expm(-tau*Ham_vj).reshape(m,m,m,m)

    expH = []

    if (Lx == 2 or Ly == 2):
        if Ly == 2 and Lx == 2:
            for i in range(N):
                expH_local = [expH_vij,expH_vij]
                expH.append(expH_local)
        elif (Ly == 2):
            for i in range(N):
                ix = i // (Ly)
                if (ix % 2 == 0):
                    iy = i % Ly
                else:
                    iy = Ly - 1 - (i % Ly)
                if iy == 0: #bottom
                    if (ix == 0): #left 
                        expH_local = [expH_vi,expH_vij]
                        expH.append(expH_local)
                    elif(ix == Lx - 2):
                        expH_local = [expH_vj,expH_eij]
                        expH.append(expH_local)
                    elif(ix == Lx - 1):
                        expH_local = [None,expH_vij]
                        expH.append(expH_local)
                    else:
                        expH_local = [expH_eij,expH_eij] 
                        expH.append(expH_local)
                else:
                    if (ix == 0):
                        expH_local = [expH_vi,None]
                        expH.append(expH_local)
                    elif(ix == Lx - 2):
                        expH_local = [expH_vj,None]
                        expH.append(expH_local)
                    elif(ix == Lx - 1):
                        expH_local = [None,None]
                        expH.append(expH_local)
                    else:
                        expH_local = [expH_eij,None]
                        expH.append(expH_local)
        else: # Lx == 2
            for i in range(N):
                ix = i // (Ly)
                if (ix % 2 == 0):
                    iy = i % Ly
                else:
                    iy = Ly - 1 - (i % Ly)
                if ix == 0: #bottom
                    if (iy == 0): #bottom
                        expH_local = [expH_vij,expH_vi]
                        expH.append(expH_local)
                    elif(iy == Ly - 2):
                        expH_local = [expH_eij,expH_vj]
                        expH.append(expH_local)
                    elif(iy == Ly - 1):
                        expH_local = [expH_vij,None]
                        expH.append(expH_local)
                    else:
                        expH_local = [expH_eij,expH_eij]
                        expH.append(expH_local)
                else:
                    if (iy == 0):
                        expH_local = [None, expH_vi]
                        expH.append(expH_local)
                    elif(iy == Ly - 2):
                        expH_local = [None, expH_vj]
                        expH.append(expH_local)
                    elif(iy == Ly - 1):
                        expH_local = [None,None]
                        expH.append(expH_local)
                    else:
                        expH_local = [None, expH_eij]
                        expH.append(expH_local)

    else:
        for i in range(N):
            ix = i // (Ly)
            if (ix % 2 == 0):
                iy = i % Ly
            else:
                iy = Ly - 1 - (i % Ly)
            if (iy == 0):
                if (ix == 0): #left edge
                    expH_local = [expH_vi,expH_vi] 
                    expH.append(expH_local)
                elif (ix == Lx - 2):
                    expH_local = [expH_vj,expH_ei] 
                    expH.append(expH_local)
                elif (ix == Lx - 1):
                    expH_local = [None, expH_vi] 
                    expH.append(expH_local)
                else:
                    expH_local = [expH_eij, expH_ei] 
                    expH.append(expH_local)
            elif (iy == Ly - 2):
                if (ix == 0): #left edge
                    expH_local = [expH_ei,expH_vj] 
                    expH.append(expH_local)
                elif (ix == Lx - 2):
                    expH_local = [expH_ej,expH_ej] 
                    expH.append(expH_local)
                elif (ix == Lx - 1):
                    expH_local = [None, expH_vj] 
                    expH.append(expH_local)
                else:
                    expH_local = [expH_b, expH_ej] 
                    expH.append(expH_local)
            elif (iy == Ly - 1):
                if (ix == 0): #left edge
                    expH_local = [expH_vi,None] 
                    expH.append(expH_local)
                elif (ix == Lx - 2):
                    expH_local = [expH_vj,None] 
                    expH.append(expH_local)
                elif (ix == Lx - 1):
                    expH_local = [None,  None] 
                    expH.append(expH_local)
                else:
                    expH_local = [expH_eij, None] 
                    expH.append(expH_local)
            else:
                if (ix == 0): #left edge
                    expH_local = [expH_ei,expH_eij] 
                    expH.append(expH_local)
                elif (ix == Lx - 2):
                    expH_local = [expH_ej,expH_b] 
                    expH.append(expH_local)
                elif (ix == Lx - 1):
                    expH_local = [None,  expH_eij] 
                    expH.append(expH_local)
                else:
                    expH_local = [expH_b, expH_b] 
                    expH.append(expH_local)
    return expH



def mult_left(v,lam_i,Tn_i):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i,(0,1)),Tn_i.conj(),([0,1],[1,0]))

def mult_right(v,lam_i,Tn_i):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i,(0,2)),Tn_i.conj(),([0,1],[2,0]))

def mult_left_op(v,lam_i,Tn_i,op):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i,(0,1)),op,(1,1)),Tn_i.conj(),([0,2],[1,0]))

def Contract_n_site_no_op(El,Er,Tn,lam,left,right):
    v = mult_left(El,lam[left],Tn[left])
    for i in range(left+1,right):
        v = mult_left(v,lam[i],Tn[i])
    return np.tensordot(mult_left(v,lam[right],Tn[right]),Er,([0,1],[0,1]))
def Contract_n_site_2op(El,Er,Tn,lam,left,right,op1,op2):
    v = mult_left_op(El,lam[left],Tn[left],op1)
    for i in range(left+1,right):
        v = mult_left(v,lam[i],Tn[i])
    return np.tensordot(mult_left_op(v,lam[right],Tn[right],op2),Er,([0,1],[0,1]))
def Calc_dot_2d(Env_left,Env_right,Tn,lam,Sz,Sp,Sm,Lx, Ly):
    N = Lx * Ly
    zz = np.zeros((N,2))
    pm = np.zeros((N,2))
    mp = np.zeros((N,2))
    ## horizontal
    for ix in range(Lx -1):
        for iy in range(Ly):
            if ix % 2 == 0:
                i = iy + ix * Ly
                j = i + 2 * (Ly - 1 - iy) + 1
            else:
                i = ix * Ly + (Ly - 1 - iy)
                j = i + 2 * iy + 1

            norm = Contract_n_site_no_op(Env_left[i],Env_right[j],Tn, lam,i, j)
            zz[i,0] = np.real(Contract_n_site_2op(Env_left[i],Env_right[j],Tn, lam,i, j,Sz, Sz)/norm)
            pm[i,0] = np.real(Contract_n_site_2op(Env_left[i],Env_right[j],Tn, lam,i, j,Sp, Sm)/norm)
            mp[i,0] = np.real(Contract_n_site_2op(Env_left[i],Env_right[j],Tn, lam,i, j,Sm, Sp)/norm)

    ## virtical
    for ix in range(Lx):
        for iy in range(Ly-1):
            if ix % 2 == 0:
                i = iy + ix * Ly
                j = (iy + 1) + ix * Ly

                # j = i + 1
                norm = Contract_two_site_no_op(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1])
                zz[i,1] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sz,Sz)/norm)
                pm[i,1] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sp,Sm)/norm)
                mp[i,1] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sm,Sp)/norm)

            else:
                i = ix * Ly + (Ly - 1 - iy)
                j = ix * Ly + (Ly - 2 - iy)

                # j = i - 1

                norm = Contract_two_site_no_op(Env_left[j],Env_right[j+1],Tn[j],Tn[j+1],lam[j],lam[j+1])
                zz[i,1] = np.real(Contract_two_site(Env_left[j],Env_right[j+1],Tn[j],Tn[j+1],lam[j],lam[j+1],Sz,Sz)/norm)
                pm[i,1] = np.real(Contract_two_site(Env_left[j],Env_right[j+1],Tn[j],Tn[j+1],lam[j],lam[j+1],Sp,Sm)/norm)
                mp[i,1] = np.real(Contract_two_site(Env_left[j],Env_right[j+1],Tn[j],Tn[j+1],lam[j],lam[j+1],Sm,Sp)/norm)
    return zz,pm,mp

def Calc_Energy_2d(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,hz,D,Lx,Ly):
    N = Lx * Ly
    m = Tn[0].shape[0]
    
    Sp = np.zeros((m,m))
    for i in range(1,m):
        Sp[i-1,i] = np.sqrt(i * (m - i))
    
    Sm = np.zeros((m,m))
    for i in range(0,m-1):
        Sm[i+1,i] = np.sqrt((i + 1.0) * (m - 1.0 - i))

    Sz = np.zeros((m,m))
    for i in range(m):
        Sz[i,i] = 0.5 * (m - 1.0) - i
    
    Sx = 0.5 * (Sp + Sm)
    Sz2 = np.dot(Sz,Sz)

    Id = np.identity(m)

    mx = np.zeros(N)
    mz = np.zeros(N)
    z2 = np.zeros(N)
    
    for i in range(N):
        norm = Contract_one_site_no_op(Env_left[i],Env_right[i],Tn[i],lam[i])
        mx[i] = np.real(Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sx)/norm)
        mz[i] = np.real(Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sz)/norm)
        z2[i] = np.real(Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sz2)/norm)
    
    zz, pm, mp = Calc_dot_2d(Env_left, Env_right, Tn, lam, Sz, Sp, Sm, Lx, Ly)
    
    E = (Jz * np.sum(zz) + 0.5 * Jxy * (np.sum(pm) + np.sum(mp)) -hx * np.sum(mx) -hz * np.sum(mz) + D * np.sum(z2)) #/ (N-1)
    return E

def TEBD_IT_Simulation_2d(m,Jz,Jxy,hx,hz,D,Lx, Ly,chi_max,tau_max,tau_min,tau_step,inv_precision=1e-10,second_ST=False,tensor_dtype=np.dtype(float),output_dyn=False,output_dyn_num=100,output=False):
    tau_factor = (tau_min/tau_max)**(1.0/tau_step)
    output_step = tau_step//output_dyn_num

    
    Ham_b = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position = 'bulk')
    Ham_eij = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position = 'edge_ij')
    Ham_ei = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position = 'edge_i')
    Ham_ej = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position = 'edge_j')
    Ham_vij = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position= 'vertex_ij')
    Ham_vi = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position= 'vertex_i')
    Ham_vj = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position= 'vertex_j')

    ## Trivial Canonical form initial condition
    Tn = []
    lam = []

    N = Lx * Ly
    for i in range(N):
        ix = i // (Ly)
        if (ix % 2 == 0):
            iy = i % Ly
        else:
            iy = Ly - 1 - (i % Ly)
        Tn.append(np.zeros((m,1,1),dtype=tensor_dtype))            
        if ( (ix + iy) % 2 == 0):
            Tn[-1][0,0,0] = 1.0
        else:
            Tn[-1][m-1,0,0] = 1.0
        lam.append(np.ones((1),dtype=float))
    lam.append(np.ones((1),dtype=float))
        
    ## Imaginary time evolution (TEBD algorithm)
    tau = tau_max
    T = 0.0
    T_list = []
    E_list = []
    mz_list = []
    S_list = []

    for n in range(tau_step):
        ## make expH list
        expH = make_expH_list(Ham_b, Ham_eij, Ham_ei, Ham_ej, Ham_vij, Ham_vi, Ham_vj, tau, Lx, Ly, m)
        if (second_ST):
            expH2 = make_expH_list(Ham_b, Ham_eij, Ham_ei, Ham_ej, Ham_vij, Ham_vi, Ham_vj, 0.5 * tau, Lx, Ly, m)
        
        if (output_dyn and n%output_step == 0):
            Env_left,Env_right = Calc_Environment(Tn,lam)
            
            mz = Calc_mag(Env_left,Env_right,Tn,lam)
            E = Calc_Energy_2d(Env_left,Env_right,Tn,lam,Jz, Jxy,hx,hz, D, Lx, Ly)
            prob = lam[N//2]**2
            S = -np.sum(prob * np.log(prob)) # Entropy
            #print(f"##Dyn {T} {E} {np.sqrt(np.sum(mz**2)/N)} {mz}")
            print(f"##Dyn {T} {E} {np.sum(mz)/N} {S}")

            T_list.append(T)
            E_list.append(E)
            mz_list.append(mz)
            S_list.append(S)
            
        if second_ST:
            Tn,lam = TEBD_update_2d_second(Tn,lam,expH,expH2,chi_max,Lx, Ly,inv_precision=inv_precision)
        else:
            Tn,lam = TEBD_update_2d(Tn,lam,expH,chi_max,Lx, Ly,inv_precision=inv_precision)

        T += tau 
        tau = tau*tau_factor

    if output:
        Env_left,Env_right = Calc_Environment(Tn,lam)        
        mz = Calc_mag(Env_left,Env_right,Tn,lam)    
        E = Calc_Energy_2d(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,hz,D,Lx,Ly)
        prob = lam[N//2]**2
        S = -np.sum(prob * np.log(prob)) # Entropy
        #print(m, Jz, Jxy, hx, hz, D, E, np.sqrt(np.sum(mz**2)/N))
        print(m, Jz, Jxy, hx, hz, D, E, np.sum(mz)/N, S)

    if output_dyn:
        return Tn,lam,np.array(T_list),np.array(E_list),np.array(mz_list),np.array(S_list)
    else:
        return Tn,lam

def TEBD_RT_Simulation_2d(m,Jz,Jxy,hx,hz,D,Lx, Ly,chi_max,dt,t_step,init_Tn, init_lam, inv_precision=1e-10,second_ST=False,tensor_dtype=np.dtype(complex),output_dyn=False,output_dyn_num=100,output=False):
    output_step = t_step//output_dyn_num

    Ham_b = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position = 'bulk')
    Ham_eij = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position = 'edge_ij')
    Ham_ei = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position = 'edge_i')
    Ham_ej = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position = 'edge_j')
    Ham_vij = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position= 'vertex_ij')
    Ham_vi = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position= 'vertex_i')
    Ham_vj = set_Hamiltonian_S_2d(m,Jz,Jxy,hx,hz,D,position= 'vertex_j')
    
    ## From initial condition
    Tn = copy.deepcopy(init_Tn)
    lam = copy.deepcopy(init_lam)
    N = Lx * Ly
        
    ## Real time evolution (TEBD algorithm)
    T = 0.0
    T_list = []
    E_list = []
    mz_list = []
    S_list = []

    ## real time evolution operator U
    expH = make_expH_list(Ham_b, Ham_eij, Ham_ei, Ham_ej, Ham_vij, Ham_vi, Ham_vj, 1.0j * dt, Lx, Ly, m)
    if (second_ST):
        expH2 = make_expH_list(Ham_b, Ham_eij, Ham_ei, Ham_ej, Ham_vij, Ham_vi, Ham_vj, 0.5j * dt, Lx, Ly, m)
    
    for n in range(t_step):
        
        if (output_dyn and n%output_step == 0):
            Env_left,Env_right = Calc_Environment(Tn,lam)
            
            mz = Calc_mag(Env_left,Env_right,Tn,lam)
            E = Calc_Energy_2d(Env_left,Env_right,Tn,lam,Jz, Jxy,hx,hz,D,Lx,Ly)
            prob = lam[N//2]**2
            S = -np.sum(prob * np.log(prob)) # Entropy
            #print(f"##Dyn {T} {E} {np.sqrt(np.sum(mz**2)/N)} {mz}")
            print(f"##Dyn {T} {E} {np.sum(mz)/N} {S}")

            T_list.append(T)
            E_list.append(E)
            mz_list.append(mz)
            S_list.append(S)
            
        if second_ST:
            Tn,lam = TEBD_update_2d_second(Tn,lam,expH,expH2,chi_max,Lx, Ly,inv_precision=inv_precision)
        else:
            Tn,lam = TEBD_update_2d(Tn,lam,expH,chi_max,Lx, Ly,inv_precision=inv_precision)

        T = (n + 1) *dt

    if output:
        Env_left,Env_right = Calc_Environment(Tn,lam)        
        mz = Calc_mag(Env_left,Env_right,Tn,lam)    
        E = Calc_Energy_2d(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,hz,D,Lx,Ly)
        prob = lam[N//2]**2
        S = -np.sum(prob * np.log(prob)) # Entropy
        #print(m, Jz, Jxy, hx, hz, D, E, np.sqrt(np.sum(mz**2)/N))
        print(m, Jz, Jxy, hx, hz, D, E, np.sum(mz)/N, S)

    if output_dyn:
        return Tn,lam,np.array(T_list),np.array(E_list),np.array(mz_list),np.array(S_list)
    else:
        return Tn,lam


#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='TEBD siumulator for one dimensional spin model')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=10,
                        help='set system size N for chain (for square lattice this N is ignored) (default = 10)')
    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=-1.0,
                        help='amplitude for SzSz interaction  (default = -1.0)')
    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=0.0,
                        help='amplitude for SxSx + SySy interaction  (default = 0.0)')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=2,
                        help='Spin size m=2S +1  (default = 2)' )
    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=0.5,
                        help='extarnal magnetix field along x (default = 0.5)')
    parser.add_argument('-hz', metavar='hz',dest='hz', type=float, default=0.0,
                        help='extarnal magnetix field along z (default = 0.0)')
    parser.add_argument('-D', metavar='D',dest='D', type=float, default=0.0,
                        help='single ion anisotropy Sz^2  (default = 0.0)')
    parser.add_argument('-chi', metavar='chi_max',dest='chi_max', type=int, default=20,
                        help='maximum bond dimension at truncation  (default = 20)')
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
    parser.add_argument('--square', action='store_const', const=True,
                        default=False, help='simulate 2d spin model on the Lx * Ly square lattice')
    parser.add_argument('-Lx', metavar='Lx',dest='Lx', type=int, default=3,
                        help='set system size Lx  for square lattice (default = 3)')
    parser.add_argument('-Ly', metavar='Ly',dest='Ly', type=int, default=3,
                        help='set system size Ly  for square lattice (default = 3)')
    return parser.parse_args()
    

if __name__ == "__main__":
    ## read params from command line
    args = parse_args()

    if args.use_complex:
        tensor_dtype = np.dtype(complex)
    else:
        tensor_dtype = np.dtype(float)
    if args.square:
        if args.output_dyn:
            Tn, lam, T_list, E_list, mz_list, S_list= TEBD_IT_Simulation_2d(args.m,args.Jz,args.Jxy,args.hx,args.hz,args.D,args.Lx, args.Ly,args.chi_max,args.tau_max,args.tau_min,args.tau_step,args.inv_precision,args.second_ST,tensor_dtype,args.output_dyn,args.output_dyn_num,output=True)
        else:
            Tn, lam = TEBD_IT_Simulation_2d(args.m,args.Jz,args.Jxy,args.hx,args.hz,args.D,args.Lx, args.Ly,args.chi_max,args.tau_max,args.tau_min,args.tau_step,args.inv_precision,args.second_ST,tensor_dtype,args.output_dyn,args.output_dyn_num,output=True)
    else:
        if args.output_dyn:
            Tn, lam, T_list, E_list, mz_list, S_list = TEBD_IT_Simulation(args.m,args.Jz,args.Jxy,args.hx,args.hz,args.D,args.N,args.chi_max,args.tau_max,args.tau_min,args.tau_step,args.inv_precision,args.second_ST,tensor_dtype,args.output_dyn,args.output_dyn_num,output=True)
        else:
            Tn, lam = TEBD_IT_Simulation(args.m,args.Jz,args.Jxy,args.hx,args.hz,args.D,args.N,args.chi_max,args.tau_max,args.tau_min,args.tau_step,args.inv_precision,args.second_ST,tensor_dtype,args.output_dyn,args.output_dyn_num,output=True)

