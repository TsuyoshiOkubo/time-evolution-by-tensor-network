import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse
import copy

def set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position='bulk'):
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

def simple_update(Tn,lam,expH,chi_max,Lx, Ly,inv_precision=1e-10,periodic=False):
    N = Lx * Ly
    m = expH[0][0].shape[0]

    lam_inv =[]

    for i in range(N):
        lam_inv_local = []
        for j in range(4):
            lam_inv_local.append(1.0/lam[i][j])
            
        lam_inv.append(lam_inv_local)

    if periodic:
        Lx_loop = Lx
        Ly_loop = Ly
    else:
        Lx_loop = Lx - 1
        Ly_loop = Ly - 1

    ## horizontal update
    for eo in range(2):
        for ix in range(Lx_loop):
            for iy in range(Ly):
                if (ix + iy) % 2 == eo:
                
                    i = ix + iy * Lx
                    if periodic:
                        j = (ix + 1) % Lx + iy * Lx
                    else:
                        j = (ix + 1) + iy * Lx
                    ## mult env
                    Tnl = Tn[i] * lam[i][0][:,None,None,None,None]
                    Tnl *= lam[i][1][None,:,None,None,None]
                    Tnl *= lam[i][3][None,None,None,:,None]

                    Tnr = Tn[j] * lam[j][1][None,:,None,None,None]
                    Tnr *= lam[j][2][None,None,:,None,None]
                    Tnr *= lam[j][3][None,None,None,:,None]
                    Ql, Rl = linalg.qr(Tnl.transpose(0,1,3,2,4).reshape(-1,Tn[i].shape[2]*Tn[i].shape[4]),mode='economic')
                    Qr, Rr = linalg.qr(Tnr.transpose(1,2,3,0,4).reshape(-1,Tn[j].shape[0]*Tn[j].shape[4]),mode='economic')

                    chi_l = Rl.shape[0]
                    chi_r = Rr.shape[0]
                    Rl = Rl.reshape(-1,Tn[i].shape[2],Tn[i].shape[4]) * lam[i][2][None,:,None]  
                    Rr = Rr.reshape(-1,Tn[j].shape[0],Tn[j].shape[4]) 

                    Theta = np.tensordot(np.tensordot(Rl,Rr,([1],[1])),expH[i][0],[[1,3],[2,3]]).transpose(0,2,1,3).reshape(chi_l*m,chi_r*m)
                    ## SVD
                    U,s,VT = linalg.svd(Theta,full_matrices=False)

                    ## Truncation
                    ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
                    chi = np.min([np.sum(s > inv_precision),chi_max])
                    lam_new = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
                    lam[i][2] = lam_new
                    lam[j][0] = lam_new
                    lam_inv[i][2] = 1.0/lam_new
                    lam_inv[j][0] = 1.0/lam_new

                    ## remove env
                    Ql = Ql.reshape(Tn[i].shape[0],Tn[i].shape[1],Tn[i].shape[3],chi_l) * lam_inv[i][0][:,None,None,None]
                    Ql *= lam_inv[i][1][None,:,None,None]  
                    Ql *= lam_inv[i][3][None,None,:,None]
                    Tn[i] = np.tensordot(Ql,U[:,:chi].reshape(chi_l,m,chi),([3],[0])).transpose(0,1,4,2,3)

                    Qr = Qr.reshape(Tn[j].shape[1],Tn[j].shape[2],Tn[j].shape[3],chi_r) * lam_inv[j][1][:,None,None,None]
                    Qr *= lam_inv[j][2][None,:,None,None]  
                    Qr *= lam_inv[j][3][None,None,:,None]
                    Tn[j] = np.tensordot(Qr,VT[:chi,:].reshape(chi,chi_r,m),([3],[1])).transpose(3,0,1,2,4)

    ## virtical update
    for eo in range(2):
        for ix in range(Lx):
            for iy in range(Ly_loop):
                if (ix + iy) % 2 == eo:
                
                    i = ix + iy * Lx
                    if periodic:
                        j = ix  + (iy + 1) % Ly  * Lx
                    else:   
                        j = ix + (iy + 1) * Lx
                    ## mult env
                    Tnl = Tn[i] * lam[i][0][:,None,None,None,None]
                    Tnl *= lam[i][2][None,None,:,None,None]
                    Tnl *= lam[i][3][None,None,None,:,None]

                    Tnr = Tn[j] * lam[j][0][:,None, None,None,None]
                    Tnr *= lam[j][1][None,:,None,None,None]
                    Tnr *= lam[j][2][None,None,:,None,None]
                    Ql, Rl = linalg.qr(Tnl.transpose(0,2,3,1,4).reshape(-1,Tn[i].shape[1]*Tn[i].shape[4]),mode='economic')
                    Qr, Rr = linalg.qr(Tnr.reshape(-1,Tn[j].shape[3]*Tn[j].shape[4]),mode='economic')

                    chi_l = Rl.shape[0]
                    chi_r = Rr.shape[0]
                    Rl = Rl.reshape(-1,Tn[i].shape[1],Tn[i].shape[4]) * lam[i][1][None,:,None]  
                    Rr = Rr.reshape(-1,Tn[j].shape[3],Tn[j].shape[4]) 

                    Theta = np.tensordot(np.tensordot(Rl,Rr,([1],[1])),expH[i][1],[[1,3],[2,3]]).transpose(0,2,1,3).reshape(chi_l*m,chi_r*m)
                    ## SVD
                    U,s,VT = linalg.svd(Theta,full_matrices=False)

                    ## Truncation
                    ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
                    chi = np.min([np.sum(s > inv_precision),chi_max])
                    lam_new = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
                    lam[i][1] = lam_new
                    lam[j][3] = lam_new
                    lam_inv[i][1] = 1.0/lam_new
                    lam_inv[j][3] = 1.0/lam_new

                    ## remove env
                    Ql = Ql.reshape(Tn[i].shape[0],Tn[i].shape[2],Tn[i].shape[3],chi_l) * lam_inv[i][0][:,None,None,None]
                    Ql *= lam_inv[i][2][None,:,None,None]  
                    Ql *= lam_inv[i][3][None,None,:,None]
                    Tn[i] = np.tensordot(Ql,U[:,:chi].reshape(chi_l,m,chi),([3],[0])).transpose(0,4,1,2,3)

                    Qr = Qr.reshape(Tn[j].shape[0],Tn[j].shape[1],Tn[j].shape[2],chi_r) * lam_inv[j][0][:,None,None,None]
                    Qr *= lam_inv[j][1][None,:,None,None]  
                    Qr *= lam_inv[j][2][None,None,:,None]
                    Tn[j] = np.tensordot(Qr,VT[:chi,:].reshape(chi,chi_r,m),([3],[1]))

    return Tn,lam

def simple_update_second(Tn,lam,expH,expH2,chi_max,Lx, Ly,inv_precision=1e-10,periodic=False):
    N = Lx * Ly
    m = expH[0][0].shape[0]

    lam_inv =[]

    for i in range(N):
        lam_inv_local = []
        for j in range(4):
            lam_inv_local.append(1.0/lam[i][j])
            
        lam_inv.append(lam_inv_local)

    if periodic:
        Lx_loop = Lx
        Ly_loop = Ly
    else:
        Lx_loop = Lx - 1
        Ly_loop = Ly - 1

    ## horizontal update
    for eo in range(2):
        for ix in range(Lx_loop):
            for iy in range(Ly):
                if (ix + iy) % 2 == eo:
                
                    i = ix + iy * Lx
                    if periodic:
                        j = (ix + 1) % Lx + iy * Lx
                    else:
                        j = (ix + 1) + iy * Lx
                    ## mult env
                    Tnl = Tn[i] * lam[i][0][:,None,None,None,None]
                    Tnl *= lam[i][1][None,:,None,None,None]
                    Tnl *= lam[i][3][None,None,None,:,None]

                    Tnr = Tn[j] * lam[j][1][None,:,None,None,None]
                    Tnr *= lam[j][2][None,None,:,None,None]
                    Tnr *= lam[j][3][None,None,None,:,None]
                    Ql, Rl = linalg.qr(Tnl.transpose(0,1,3,2,4).reshape(-1,Tn[i].shape[2]*Tn[i].shape[4]),mode='economic')
                    Qr, Rr = linalg.qr(Tnr.transpose(1,2,3,0,4).reshape(-1,Tn[j].shape[0]*Tn[j].shape[4]),mode='economic')

                    chi_l = Rl.shape[0]
                    chi_r = Rr.shape[0]
                    Rl = Rl.reshape(-1,Tn[i].shape[2],Tn[i].shape[4]) * lam[i][2][None,:,None]  
                    Rr = Rr.reshape(-1,Tn[j].shape[0],Tn[j].shape[4])

                    Theta = np.tensordot(np.tensordot(Rl,Rr,([1],[1])),expH2[i][0],[[1,3],[2,3]]).transpose(0,2,1,3).reshape(chi_l*m,chi_r*m)
                    ## SVD
                    U,s,VT = linalg.svd(Theta,full_matrices=False)

                    ## Truncation
                    ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
                    chi = np.min([np.sum(s > inv_precision),chi_max])
                    lam_new = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
                    lam[i][2] = lam_new
                    lam[j][0] = lam_new
                    lam_inv[i][2] = 1.0/lam_new
                    lam_inv[j][0] = 1.0/lam_new

                    ## remove env
                    Ql = Ql.reshape(Tn[i].shape[0],Tn[i].shape[1],Tn[i].shape[3],chi_l) * lam_inv[i][0][:,None,None,None]
                    Ql *= lam_inv[i][1][None,:,None,None]  
                    Ql *= lam_inv[i][3][None,None,:,None]
                    Tn[i] = np.tensordot(Ql,U[:,:chi].reshape(chi_l,m,chi),([3],[0])).transpose(0,1,4,2,3)

                    Qr = Qr.reshape(Tn[j].shape[1],Tn[j].shape[2],Tn[j].shape[3],chi_r) * lam_inv[j][1][:,None,None,None]
                    Qr *= lam_inv[j][2][None,:,None,None]  
                    Qr *= lam_inv[j][3][None,None,:,None]
                    Tn[j] = np.tensordot(Qr,VT[:chi,:].reshape(chi,chi_r,m),([3],[1])).transpose(3,0,1,2,4)

    ## virtical update
    for eoe in range(3):
        eo = eoe % 2
        for ix in range(Lx):
            for iy in range(Ly_loop):
                if (ix + iy) % 2 == eo:
                
                    i = ix + iy * Lx
                    if periodic:
                        j = ix  + (iy + 1) % Ly  * Lx
                    else:
                        j = ix + (iy + 1) * Lx
                    ## mult env
                    Tnl = Tn[i] * lam[i][0][:,None,None,None,None]
                    Tnl *= lam[i][2][None,None,:,None,None]
                    Tnl *= lam[i][3][None,None,None,:,None]

                    Tnr = Tn[j] * lam[j][0][:,None,None,None,None]
                    Tnr *= lam[j][1][None,:,None,None,None]
                    Tnr *= lam[j][2][None,None,:,None,None]
                    Ql, Rl = linalg.qr(Tnl.transpose(0,2,3,1,4).reshape(-1,Tn[i].shape[1]*Tn[i].shape[4]),mode='economic')
                    Qr, Rr = linalg.qr(Tnr.reshape(-1,Tn[j].shape[3]*Tn[j].shape[4]),mode='economic')

                    chi_l = Rl.shape[0]
                    chi_r = Rr.shape[0]
                    Rl = Rl.reshape(-1,Tn[i].shape[1],Tn[i].shape[4]) * lam[i][1][None,:,None]  
                    Rr = Rr.reshape(-1,Tn[j].shape[3],Tn[j].shape[4])

                    if eo == 0:
                        # 0.5 dt
                        Theta = np.tensordot(np.tensordot(Rl,Rr,([1],[1])),expH2[i][1],[[1,3],[2,3]]).transpose(0,2,1,3).reshape(chi_l*m,chi_r*m)
                    else:
                        # dt
                        Theta = np.tensordot(np.tensordot(Rl,Rr,([1],[1])),expH[i][1],[[1,3],[2,3]]).transpose(0,2,1,3).reshape(chi_l*m,chi_r*m)
                    ## SVD
                    U,s,VT = linalg.svd(Theta,full_matrices=False)

                    ## Truncation
                    ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
                    chi = np.min([np.sum(s > inv_precision),chi_max])
                    lam_new = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
                    lam[i][1] = lam_new
                    lam[j][3] = lam_new
                    lam_inv[i][1] = 1.0/lam_new
                    lam_inv[j][3] = 1.0/lam_new

                    ## remove env
                    Ql = Ql.reshape(Tn[i].shape[0],Tn[i].shape[2],Tn[i].shape[3],chi_l) * lam_inv[i][0][:,None,None,None]
                    Ql *= lam_inv[i][2][None,:,None,None]  
                    Ql *= lam_inv[i][3][None,None,:,None]
                    Tn[i] = np.tensordot(Ql,U[:,:chi].reshape(chi_l,m,chi),([3],[0])).transpose(0,4,1,2,3)

                    Qr = Qr.reshape(Tn[j].shape[0],Tn[j].shape[1],Tn[j].shape[2],chi_r) * lam_inv[j][0][:,None,None,None]
                    Qr *= lam_inv[j][1][None,:,None,None]  
                    Qr *= lam_inv[j][2][None,None,:,None]
                    Tn[j] = np.tensordot(Qr,VT[:chi,:].reshape(chi,chi_r,m),([3],[1]))

    ## horizontal update
    for eo in (1,0):
        for ix in range(Lx_loop):
            for iy in range(Ly):
                if (ix + iy) % 2 == eo:
                
                    i = ix + iy * Lx
                    if periodic:
                        j = (ix + 1) % Lx + iy * Lx
                    else:
                        j = (ix + 1) + iy * Lx
                    ## mult env
                    Tnl = Tn[i] * lam[i][0][:,None,None,None,None]
                    Tnl *= lam[i][1][None,:,None,None,None]
                    Tnl *= lam[i][3][None,None,None,:,None]

                    Tnr = Tn[j] * lam[j][1][None,:,None,None,None]
                    Tnr *= lam[j][2][None,None,:,None,None]
                    Tnr *= lam[j][3][None,None,None,:,None]
                    Ql, Rl = linalg.qr(Tnl.transpose(0,1,3,2,4).reshape(-1,Tn[i].shape[2]*Tn[i].shape[4]),mode='economic')
                    Qr, Rr = linalg.qr(Tnr.transpose(1,2,3,0,4).reshape(-1,Tn[j].shape[0]*Tn[j].shape[4]),mode='economic')

                    chi_l = Rl.shape[0]
                    chi_r = Rr.shape[0]
                    Rl = Rl.reshape(-1,Tn[i].shape[2],Tn[i].shape[4]) * lam[i][2][None,:,None]  
                    Rr = Rr.reshape(-1,Tn[j].shape[0],Tn[j].shape[4])

                    Theta = np.tensordot(np.tensordot(Rl,Rr,([1],[1])),expH2[i][0],[[1,3],[2,3]]).transpose(0,2,1,3).reshape(chi_l*m,chi_r*m)
                    ## SVD
                    U,s,VT = linalg.svd(Theta,full_matrices=False)

                    ## Truncation
                    ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
                    chi = np.min([np.sum(s > inv_precision),chi_max])
                    lam_new = s[:chi]/np.sqrt(np.sum(s[:chi]**2))
                    lam[i][2] = lam_new
                    lam[j][0] = lam_new
                    lam_inv[i][2] = 1.0/lam_new
                    lam_inv[j][0] = 1.0/lam_new

                    ## remove env
                    Ql = Ql.reshape(Tn[i].shape[0],Tn[i].shape[1],Tn[i].shape[3],chi_l) * lam_inv[i][0][:,None,None,None]
                    Ql *= lam_inv[i][1][None,:,None,None]  
                    Ql *= lam_inv[i][3][None,None,:,None]
                    Tn[i] = np.tensordot(Ql,U[:,:chi].reshape(chi_l,m,chi),([3],[0])).transpose(0,1,4,2,3)

                    Qr = Qr.reshape(Tn[j].shape[1],Tn[j].shape[2],Tn[j].shape[3],chi_r) * lam_inv[j][1][:,None,None,None]
                    Qr *= lam_inv[j][2][None,:,None,None]  
                    Qr *= lam_inv[j][3][None,None,:,None]
                    Tn[j] = np.tensordot(Qr,VT[:chi,:].reshape(chi,chi_r,m),([3],[1])).transpose(3,0,1,2,4)
    

    return Tn, lam


def Contract_one_site(Tn_i,lam_i,op):
    Tn_i_temp = Tn_i * lam_i[0][:,None,None,None,None]
    Tn_i_temp *= lam_i[1][None,:,None,None,None]
    Tn_i_temp *= lam_i[2][None,None,:,None,None]
    Tn_i_temp *= lam_i[3][None,None,None,:,None]
    
    return np.vdot(Tn_i_temp.ravel(), np.tensordot(Tn_i_temp, op, ([4],[1])).ravel())

def Contract_one_site_no_op(Tn_i,lam_i):
    Tn_i_temp = Tn_i * lam_i[0][:,None,None,None,None]
    Tn_i_temp *= lam_i[1][None,:,None,None,None]
    Tn_i_temp *= lam_i[2][None,None,:,None,None]
    Tn_i_temp *= lam_i[3][None,None,None,:,None]
    
    return np.vdot(Tn_i_temp.ravel(), Tn_i_temp.ravel())


def Contract_two_site(Tn_i, Tn_j, lam_i, lam_j, op1, op2, dir='horizontal'):
    if dir == 'virtical':
        Tn_i_temp = Tn_i * lam_i[0][:,None,None,None,None]
        Tn_i_temp *= lam_i[1][None,:,None,None,None]
        Tn_i_temp *= lam_i[2][None,None,:,None,None]
        Tn_i_temp *= lam_i[3][None,None,None,:,None]
        Tn_j_temp = Tn_j * lam_j[0][:,None,None,None,None]
        Tn_j_temp *= lam_j[1][None,:,None,None,None]
        Tn_j_temp *= lam_j[2][None,None,:,None,None]

        Tn_i_temp = np.tensordot(Tn_i_temp.conj(), np.tensordot(Tn_i_temp,op1,([4],[1])),([0,2,3,4],[0,2,3,4]))
        Tn_j_temp = np.tensordot(Tn_j_temp.conj(), np.tensordot(Tn_j_temp,op2,([4],[1])),([0,1,2,4],[0,1,2,4]))

        return np.dot(Tn_i_temp.ravel(),Tn_j_temp.ravel())

    else: # horizontal
        Tn_i_temp = Tn_i * lam_i[0][:,None,None,None,None]
        Tn_i_temp *= lam_i[1][None,:,None,None,None]
        Tn_i_temp *= lam_i[2][None,None,:,None,None]
        Tn_i_temp *= lam_i[3][None,None,None,:,None]
        Tn_j_temp = Tn_j * lam_j[1][None,:,None,None,None]
        Tn_j_temp *= lam_j[2][None,None,:,None,None]
        Tn_j_temp *= lam_j[3][None,None,None,:,None]

        Tn_i_temp = np.tensordot(Tn_i_temp.conj(), np.tensordot(Tn_i_temp,op1,([4],[1])),([0,1,3,4],[0,1,3,4]))
        Tn_j_temp = np.tensordot(Tn_j_temp.conj(), np.tensordot(Tn_j_temp,op2,([4],[1])),([1,2,3,4],[1,2,3,4]))

        return np.dot(Tn_i_temp.ravel(),Tn_j_temp.ravel())



def Contract_two_site_no_op(Tn_i,Tn_j,lam_i,lam_j,dir='horizontal'):
    if dir == 'virtical':
        Tn_i_temp = Tn_i * lam_i[0][:,None,None,None,None]
        Tn_i_temp *= lam_i[1][None,:,None,None,None]
        Tn_i_temp *= lam_i[2][None,None,:,None,None]
        Tn_i_temp *= lam_i[3][None,None,None,:,None]
        Tn_j_temp = Tn_j * lam_j[0][:,None,None,None,None]
        Tn_j_temp *= lam_j[1][None,:,None,None,None]
        Tn_j_temp *= lam_j[2][None,None,:,None,None]

        Tn_i_temp = np.tensordot(Tn_i_temp.conj(), Tn_i_temp,([0,2,3,4],[0,2,3,4]))
        Tn_j_temp = np.tensordot(Tn_j_temp.conj(), Tn_j_temp,([0,1,2,4],[0,1,2,4]))

        return np.dot(Tn_i_temp.ravel(),Tn_j_temp.ravel())

    else: # horizontal
        Tn_i_temp = Tn_i * lam_i[0][:,None,None,None,None]
        Tn_i_temp *= lam_i[1][None,:,None,None,None]
        Tn_i_temp *= lam_i[2][None,None,:,None,None]
        Tn_i_temp *= lam_i[3][None,None,None,:,None]
        Tn_j_temp = Tn_j * lam_j[1][None,:,None,None,None]
        Tn_j_temp *= lam_j[2][None,None,:,None,None]
        Tn_j_temp *= lam_j[3][None,None,None,:,None]

        Tn_i_temp = np.tensordot(Tn_i_temp.conj(), Tn_i_temp,([0,1,3,4],[0,1,3,4]))
        Tn_j_temp = np.tensordot(Tn_j_temp.conj(), Tn_j_temp,([1,2,3,4],[1,2,3,4]))

        return np.dot(Tn_i_temp.ravel(),Tn_j_temp.ravel())

def Calc_mag(Tn,lam):

    N = len(Tn)
    m = Tn[0].shape[4]
    Sz = np.zeros((m,m))
    for i in range(m):
        Sz[i,i] = 0.5 * (m - 1.0) - i    
    
    mz = np.zeros(N)
    for i in range(N):
        mz[i]=np.real(Contract_one_site(Tn[i],lam[i],Sz)/Contract_one_site_no_op(Tn[i],lam[i]))
    return mz

def Calc_dot(Tn,lam,Sz,Sp,Sm, Lx, Ly,periodic=False):
    N = Lx * Ly
    zz = np.zeros((N,2))
    pm = np.zeros((N,2))
    mp = np.zeros((N,2))

    if periodic:
        Lx_loop = Lx
        Ly_loop = Ly
    else:
        Lx_loop = Lx - 1
        Ly_loop = Ly - 1
    ## horizontal
    for ix in range(Lx_loop):
        for iy in range(Ly):
            i = ix + iy * Lx
            if periodic:
                j = (ix + 1) % Lx + iy * Lx
            else:
                j = (ix + 1) + iy * Lx
            norm = Contract_two_site_no_op(Tn[i],Tn[j],lam[i],lam[j])
            zz[i,0] = np.real(Contract_two_site(Tn[i],Tn[j],lam[i],lam[j],Sz,Sz)/norm)
            pm[i,0] = np.real(Contract_two_site(Tn[i],Tn[j],lam[i],lam[j],Sp,Sm)/norm)
            mp[i,0] = np.real(Contract_two_site(Tn[i],Tn[j],lam[i],lam[j],Sm,Sp)/norm)

    ## virtical
    for ix in range(Lx):
        for iy in range(Ly_loop):
            i = ix + iy * Lx
            if periodic:
                j = ix  + (iy + 1)%Ly  * Lx
            else:
                j = ix  + (iy + 1)  * Lx

            norm = Contract_two_site_no_op(Tn[i],Tn[j],lam[i],lam[j],"virtical")
            zz[i,1] = np.real(Contract_two_site(Tn[i],Tn[j],lam[i],lam[j],Sz,Sz,"virtical")/norm)
            pm[i,1] = np.real(Contract_two_site(Tn[i],Tn[j],lam[i],lam[j],Sp,Sm,"virtical")/norm)
            mp[i,1] = np.real(Contract_two_site(Tn[i],Tn[j],lam[i],lam[j],Sm,Sp,"virtical")/norm)
    return zz,pm,mp

def Calc_Energy(Tn,lam,Jz,Jxy,hx,hz,D, Lx, Ly,periodic=False):

    N = Lx * Ly
    m = Tn[0].shape[4]
    
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

        norm = Contract_one_site_no_op(Tn[i],lam[i])
        mx[i] = np.real(Contract_one_site(Tn[i],lam[i],Sx)/norm)
        mz[i] = np.real(Contract_one_site(Tn[i],lam[i],Sz)/norm)
        z2[i] = np.real(Contract_one_site(Tn[i],lam[i],Sz2)/norm)
    
    zz, pm, mp = Calc_dot(Tn, lam, Sz, Sp, Sm, Lx, Ly, periodic)
    
    E = (Jz * np.sum(zz) + 0.5 * Jxy * (np.sum(pm) + np.sum(mp)) -hx * np.sum(mx) -hz * np.sum(mz) + D * np.sum(z2)) 
    return E

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
                ix = i % Lx
                iy = i // Lx
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
                ix = i % Lx
                iy = i // Lx
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
            ix = i % Lx
            iy = i // Lx
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


def simple_IT_Simulation(m,Jz,Jxy,hx,hz,D,Lx,Ly,chi_max,tau_max,tau_min,tau_step,inv_precision=1e-10,second_ST=False,tensor_dtype=np.dtype(float),output_dyn=False,output_dyn_num=100,output=False):

    tau_factor = (tau_min/tau_max)**(1.0/tau_step)
    output_step = tau_step//output_dyn_num
    
    Ham_b = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position = 'bulk')
    Ham_eij = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position = 'edge_ij')
    Ham_ei = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position = 'edge_i')
    Ham_ej = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position = 'edge_j')
    Ham_vij = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position= 'vertex_ij')
    Ham_vi = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position= 'vertex_i')
    Ham_vj = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position= 'vertex_j')
    
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
        expH = make_expH_list(Ham_b, Ham_eij, Ham_ei, Ham_ej, Ham_vij, Ham_vi, Ham_vj, tau, Lx, Ly, m)
        if (second_ST):
            expH2 = make_expH_list(Ham_b, Ham_eij, Ham_ei, Ham_ej, Ham_vij, Ham_vi, Ham_vj, 0.5 * tau, Lx, Ly, m)
        
        if (output_dyn and n%output_step == 0):
            
            mz = Calc_mag(Tn, lam)
            E = Calc_Energy(Tn, lam,Jz, Jxy,hx,hz, D, Lx, Ly)
            #print(f"##Dyn {T} {E} {np.sqrt(np.sum(mz**2)/N)} {mz}")
            print(f"##Dyn {T} {E} {np.sum(mz)/N}")

            T_list.append(T)
            E_list.append(E)
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
        #print(m, Jz, Jxy, hx, hz, D, E, np.sqrt(np.sum(mz**2)/N))
        print(m, Jz, Jxy, hx, hz, D, E, np.sum(mz)/N)

    if output_dyn:
        return Tn,lam,np.array(T_list),np.array(E_list),np.array(mz_list)
    else:
        return Tn,lam

def simple_RT_Simulation(m,Jz,Jxy,hx,hz,D,Lx, Ly,chi_max,dt,t_step,init_Tn, init_lam, inv_precision=1e-10,second_ST=False,tensor_dtype=np.dtype(complex),output_dyn=False,output_dyn_num=100,output=False):
    output_step = t_step//output_dyn_num

    Ham_b = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position = 'bulk')
    Ham_eij = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position = 'edge_ij')
    Ham_ei = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position = 'edge_i')
    Ham_ej = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position = 'edge_j')
    Ham_vij = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position= 'vertex_ij')
    Ham_vi = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position= 'vertex_i')
    Ham_vj = set_Hamiltonian_S(m,Jz,Jxy,hx,hz,D,position= 'vertex_j')

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
        expH = make_expH_list(Ham_b, Ham_eij, Ham_ei, Ham_ej, Ham_vij, Ham_vi, Ham_vj, 1.0j * dt, Lx, Ly, m)

        if (second_ST):
            expH2 = make_expH_list(Ham_b, Ham_eij, Ham_ei, Ham_ej, Ham_vij, Ham_vi, Ham_vj, 0.5j * dt, Lx, Ly, m)
        
        if (output_dyn and n%output_step == 0):
            mz = Calc_mag(Tn, lam)
            E = Calc_Energy(Tn, lam,Jz, Jxy,hx,hz, D, Lx, Ly)
            #print(f"##Dyn {T} {E} {np.sqrt(np.sum(mz**2)/N)} {mz}")
            print(f"##Dyn {T} {E} {np.sum(mz**2)/N}")
            T_list.append(T)
            E_list.append(E)
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
        print(m, Jz, Jxy, hx, hz, D, E, np.sum(mz)/N)

    if output_dyn:
        return Tn,lam,np.array(T_list),np.array(E_list),np.array(mz_list)
    else:
        return Tn,lam

#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='simple update siumulator for 2d spin model')
    parser.add_argument('-Lx', metavar='Lx',dest='Lx', type=int, default=4,
                        help='set system size Lx (default = 4)')
    parser.add_argument('-Ly', metavar='Ly',dest='Ly', type=int, default=4,
                        help='set system size Ly (default = 4)')
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

