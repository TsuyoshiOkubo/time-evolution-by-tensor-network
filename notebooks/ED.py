# Exact diagonalization for spin chain and square lattice
# 2017 Aug Tsuyoshi Okubo
# 2018 Dec updated to use matrix-vector multiplication
# 2025 Jul updated for supporting square lattice

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as spr
import scipy.sparse.linalg as spr_linalg
import argparse

class Hamiltonian:
    def __init__(self,m,Jz,Jxy,hx,hz,D,N,periodic=False):
        self.Jz = Jz
        self.Jxy = Jxy
        self.hx = hx
        self.hz = hz
        self.D = D
        self.N = N
        self.periodic = periodic
        self.m = m
        v_shape = (m,)
        for i in range(1,N):
            v_shape += (m,)
        self.v_shape = v_shape
        
        Sp = np.zeros((m,m))
        for i in range(1,m):
            Sp[i-1,i] = np.sqrt(i * (m - i))

        Sm = np.zeros((m,m))
        for i in range(0,m-1):
            Sm[i+1,i] = np.sqrt((i + 1.0) * (m - 1.0 - i))

        Sz = np.zeros((m,m))
        for i in range(m):
            Sz[i,i] = 0.5 * (m - 1.0) - i
        
        
        Id = np.identity(m)
        Sx = 0.5 * (Sp + Sm)
        Sz2 = np.dot(Sz,Sz)

        self.Sx = Sx
        self.Sz = Sz
        if self.periodic:
            self.pair_operator = (Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) +np.kron(Sm,Sp))
                                  - 0.5 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) 
                                  - 0.5 * hz * (np.kron(Sz,Id) + np.kron(Id,Sz))
                                  + 0.5 * D * (np.kron(Sz2,Id) + np.kron(Id,Sz2))).reshape(m,m,m,m)
            self.periodic_pair_operator = self.pair_operator
        else:
            self.pair_operator = (Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) +np.kron(Sm,Sp))
                                  - 0.5 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) 
                                  - 0.5 * hz * (np.kron(Sz,Id) + np.kron(Id,Sz))
                                  + 0.5 * D * (np.kron(Sz2,Id) + np.kron(Id,Sz2))).reshape(m,m,m,m)
            self.periodic_pair_operator = (
                                  - 0.5 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) 
                                  - 0.5 * hz * (np.kron(Sz,Id) + np.kron(Id,Sz))
                                  + 0.5 * D * (np.kron(Sz2,Id) + np.kron(Id,Sz2))).reshape(m,m,m,m)
            
                            
        ## shape of transepose after applying pair operators
        self.pair_transpose_list = []
        self.single_transpose_list = []
        for i in range(0,N-1):
            self.pair_transpose_list.append(tuple(np.arange(0,i,dtype=int)) + (N-2,N-1) + tuple(np.arange(i,N-2,dtype=int)))
        ## for N-1 (periodic boundary)
        self.pair_transpose_list.append((N-1,) + tuple(np.arange(0,N-2,dtype=int)) + (N-2,))

        for i in range(N):
            self.single_transpose_list.append(tuple(np.arange(0,i,dtype=int)) + (N-1,) + tuple(np.arange(i,N-1,dtype=int)))
    def mult_Hamiltonian(self,v):
        x = np.zeros(self.v_shape,dtype=v.dtype)
        vr = v.reshape(self.v_shape)
        
        for i in range(self.N - 1):
            x += np.tensordot(vr,self.pair_operator,axes=([i,i+1],[2,3])).transpose(self.pair_transpose_list[i])
        x += np.tensordot(vr,self.periodic_pair_operator,axes=([self.N-1,0],[2,3])).transpose(self.pair_transpose_list[self.N-1])                     
        return x.reshape(self.m**self.N)

    def mult_total_Sz(self,v):
        x = np.zeros(self.v_shape,dtype=v.dtype)
        vr = v.reshape(self.v_shape)
        
        for i in range(self.N):
            x += np.tensordot(vr,self.Sz,axes=([i],[1])).transpose(self.single_transpose_list[i])
        return x.reshape(self.m**self.N)

    def mult_total_Sx(self,v):
        x = np.zeros(self.v_shape,dtype=v.dtype)
        vr = v.reshape(self.v_shape)
        
        for i in range(self.N):
            x += np.tensordot(vr,self.Sx,axes=([i],[1])).transpose(self.single_transpose_list[i])
        return x.reshape(self.m**self.N)

class Hamiltonian_2d:
    def __init__(self,m,Jz,Jxy,hx,hz,D,Lx, Ly,periodic_x=False, periodic_y=False):
        self.Jz = Jz
        self.Jxy = Jxy
        self.hx = hx
        self.hz = hz
        self.D = D
        self.Lx = Lx
        self.Ly = Ly
        N = Lx * Ly
        self.N = N
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.m = m
        v_shape = (m,)
        for i in range(1,N):
            v_shape += (m,)
        self.v_shape = v_shape
        
        Sp = np.zeros((m,m))
        for i in range(1,m):
            Sp[i-1,i] = np.sqrt(i * (m - i))

        Sm = np.zeros((m,m))
        for i in range(0,m-1):
            Sm[i+1,i] = np.sqrt((i + 1.0) * (m - 1.0 - i))

        Sz = np.zeros((m,m))
        for i in range(m):
            Sz[i,i] = 0.5 * (m - 1.0) - i
        
        
        Id = np.identity(m)
        Sx = 0.5 * (Sp + Sm)
        Sz2 = np.dot(Sz,Sz)

        self.Sx = Sx
        self.Sz = Sz

        if self.periodic_x:
            self.pair_operator_x = (Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) +np.kron(Sm,Sp))
                                  - 0.25 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) 
                                  - 0.25 * hz * (np.kron(Sz,Id) + np.kron(Id,Sz))
                                  + 0.25 * D * (np.kron(Sz2,Id)+ np.kron(Id,Sz2))).reshape(m,m,m,m)
            self.periodic_pair_operator_x = self.pair_operator_x
        else:
            self.pair_operator_x = (Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) +np.kron(Sm,Sp))
                                  - 0.25 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) 
                                  - 0.25 * hz * (np.kron(Sz,Id) + np.kron(Id,Sz))
                                  + 0.25 * D * (np.kron(Sz2,Id)+ np.kron(Id,Sz2))).reshape(m,m,m,m)
            self.periodic_pair_operator_x = (
                                  - 0.25 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) 
                                  - 0.25 * hz * (np.kron(Sz,Id) + np.kron(Id,Sz))
                                  + 0.25 * D * (np.kron(Sz2,Id)+ np.kron(Id,Sz2))).reshape(m,m,m,m)

        if self.periodic_y:
            self.pair_operator_y = (Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) +np.kron(Sm,Sp))
                                  - 0.25 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) 
                                  - 0.25 * hz * (np.kron(Sz,Id) + np.kron(Id,Sz))
                                  + 0.25 * D * (np.kron(Sz2,Id)+ np.kron(Id,Sz2))).reshape(m,m,m,m)
            self.periodic_pair_operator_y = self.pair_operator_y
        else:
            self.pair_operator_y = (Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) +np.kron(Sm,Sp))
                                  - 0.25 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) 
                                  - 0.25 * hz * (np.kron(Sz,Id) + np.kron(Id,Sz))
                                  + 0.25 * D * (np.kron(Sz2,Id)+ np.kron(Id,Sz2))).reshape(m,m,m,m)
            self.periodic_pair_operator_y = (
                                  - 0.25 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) 
                                  - 0.25 * hz * (np.kron(Sz,Id) + np.kron(Id,Sz))
                                  + 0.25 * D * (np.kron(Sz2,Id)+ np.kron(Id,Sz2))).reshape(m,m,m,m)

                            
        ## shape of transepose after applying pair operators
        self.pair_transpose_list_x = []
        self.pair_transpose_list_y = []
        self.single_transpose_list = []
        for i in range(N):
            ix = i % Lx
            iy = i // Lx
            ## x direction
            if ix < Lx - 1:
                self.pair_transpose_list_x.append(tuple(np.arange(0,i,dtype=int)) + (N-2,N-1) + tuple(np.arange(i,N-2,dtype=int)))
            else:
                self.pair_transpose_list_x.append(tuple(np.arange(0,i-(Lx-1),dtype=int)) + (N-1,) + tuple(np.arange(i - (Lx-1),i-1,dtype=int)) + (N-2,) + tuple(np.arange(i-1,N-2,dtype=int)))
            ## y direction
            if iy < Ly- 1:
                self.pair_transpose_list_y.append(tuple(np.arange(0,i,dtype=int)) + (N-2,) + tuple(np.arange(i, i + Lx - 1,dtype=int))+ (N-1, ) + tuple(np.arange(i + Lx - 1,N-2,dtype=int)))
            else:
                self.pair_transpose_list_y.append(tuple(np.arange(0,i - (Ly-1) * Lx,dtype=int)) + (N-1,) + tuple(np.arange(i - (Ly-1) * Lx, i - 1,dtype=int)) + (N-2,) + tuple(np.arange(i-1,N-2,dtype=int)))

            self.single_transpose_list.append(tuple(np.arange(0,i,dtype=int)) + (N-1,) + tuple(np.arange(i,N-1,dtype=int)))


    def mult_Hamiltonian(self,v):
        x = np.zeros(self.v_shape,dtype=v.dtype)
        vr = v.reshape(self.v_shape)
        
        for i in range(self.N):
            ix = i % self.Lx
            iy = i // self.Lx
            ## x direction
            if ix < self.Lx - 1:            
                x += np.tensordot(vr,self.pair_operator_x,axes=([i,i+1],[2,3])).transpose(self.pair_transpose_list_x[i])
            else:
                x += np.tensordot(vr,self.periodic_pair_operator_x,axes=([i,i - (self.Lx - 1)],[2,3])).transpose(self.pair_transpose_list_x[i])
            ## y direction
            if iy < self.Ly- 1:
                 x += np.tensordot(vr,self.pair_operator_y,axes=([i, i + self.Lx],[2,3])).transpose(self.pair_transpose_list_y[i])
            else:
                 x += np.tensordot(vr,self.periodic_pair_operator_y,axes=([i, i - (self.Ly - 1) * self.Lx],[2,3])).transpose(self.pair_transpose_list_y[i])
        return x.reshape(self.m**self.N)

    def mult_total_Sz(self,v):
        x = np.zeros(self.v_shape,dtype=v.dtype)
        vr = v.reshape(self.v_shape)
        
        for i in range(self.N):
            x += np.tensordot(vr,self.Sz,axes=([i],[1])).transpose(self.single_transpose_list[i])
        return x.reshape(self.m**self.N)

    def mult_total_Sx(self,v):
        x = np.zeros(self.v_shape,dtype=v.dtype)
        vr = v.reshape(self.v_shape)
        
        for i in range(self.N):
            x += np.tensordot(vr,self.Sx,axes=([i],[1])).transpose(self.single_transpose_list[i])
        return x.reshape(self.m**self.N)


def Calc_GS(m,Jz,Jxy,hx,hz,D,N,k=5,periodic=False):
    hamiltonian = Hamiltonian(m,Jz,Jxy,hx,hz,D,N,periodic)
    Ham = spr_linalg.LinearOperator((m**N,m**N),hamiltonian.mult_Hamiltonian,dtype=float)
    eig_val,eig_vec = spr_linalg.eigsh(Ham,k=k,which="SA")
    return eig_val,eig_vec

def Calc_GS_2d(m,Jz,Jxy,hx,hz,D,Lx, Ly,k=5,periodic_x=False, periodic_y=False):
    hamiltonian = Hamiltonian_2d(m,Jz,Jxy,hx,hz,D,Lx, Ly,periodic_x, periodic_y)
    N = Lx * Ly
    Ham = spr_linalg.LinearOperator((m**N,m**N),hamiltonian.mult_Hamiltonian,dtype=float)
    eig_val,eig_vec = spr_linalg.eigsh(Ham,k=k,which="SA")
    return eig_val,eig_vec



def Calc_TE(initv, m,Jz,Jxy,hx,hz, D,N,t, periodic=False):
    hamiltonian = Hamiltonian(m,Jz,Jxy,hx,hz, D,N,periodic)
    Ham = spr_linalg.LinearOperator((m**N,m**N),matvec =lambda x: -1.0j * t * hamiltonian.mult_Hamiltonian(x), rmatvec =lambda x: 1.0j * t * hamiltonian.mult_Hamiltonian(x), dtype=complex)
    vecs = spr_linalg.expm_multiply(Ham,initv, traceA=N*t)
    return vecs

def Calc_TE_2d(initv, m,Jz,Jxy,hx,hz,D,Lx, Ly, t,periodic_x=False, periodic_y=False):
    hamiltonian = Hamiltonian_2d(m,Jz,Jxy,hx,hz,D,Lx, Ly,periodic_x, periodic_y)
    N = Lx * Ly
    Ham = spr_linalg.LinearOperator((m**N,m**N),matvec =lambda x: -1.0j * t * hamiltonian.mult_Hamiltonian(x), rmatvec =lambda x: 1.0j * t * hamiltonian.mult_Hamiltonian(x),dtype=complex)
    vecs = spr_linalg.expm_multiply(Ham,initv, traceA=N*t)
    return vecs

def Calc_ITE(initv, m,Jz,Jxy,hx,hz,D,N,t, periodic=False):
    hamiltonian = Hamiltonian(m,Jz,Jxy,hx,hz,D,N,periodic)
    Ham = spr_linalg.LinearOperator((m**N,m**N),matvec =lambda x: -t * hamiltonian.mult_Hamiltonian(x), rmatvec =lambda x: -t * hamiltonian.mult_Hamiltonian(x),dtype=initv.dtype)
    vecs = spr_linalg.expm_multiply(Ham,initv, traceA=N*t)
    return vecs

def Calc_ITE_2d(initv, m,Jz,Jxy,hx,hz,D,Lx, Ly, t,periodic_x=False, periodic_y=False):
    hamiltonian = Hamiltonian_2d(m,Jz,Jxy,hx,hz,D,Lx, Ly,periodic_x, periodic_y)
    N = Lx * Ly
    Ham = spr_linalg.LinearOperator((m**N,m**N),matvec =lambda x: -t * hamiltonian.mult_Hamiltonian(x), rmatvec =lambda x: -t * hamiltonian.mult_Hamiltonian(x),dtype=initv.dtype)
    vecs = spr_linalg.expm_multiply(Ham,initv, traceA=N*t)
    return vecs

#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='ED simulator for spin model on 1d and 2d lattices')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=10,
                        help='set system size N  for chain (for square lattice this N is ignored) (default = 10)')
    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=-1.0,
                        help='interaction for SzSz  (default = -1.0)')
    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=0.0,
                        help='interaction for SxSx + SySy  (default = 0.0)')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=2,
                        help='Spin size m=2S +1  (default = 2)')
    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=0.5,
                        help='extarnal magnetix field  (default = 0.5)')
    parser.add_argument('-hz', metavar='hz',dest='hz', type=float, default=0.0,
                        help='extarnal magnetix field  (default = 0.0)')
    parser.add_argument('-D', metavar='D',dest='D', type=float, default=0.0,
                        help='single ion anisotropy Sz^2  (default = 0.0)')
    parser.add_argument('-e_num', metavar='e_num',dest='e_num', type=int, default=5,
                        help='number of calculating energies (default = 5)')
    parser.add_argument('--periodic',dest='periodic', action="store_true",
                        help='set periodic boundary condision for chain or x direction (default = open)')
    parser.add_argument('--square',dest='calc_square', action="store_true",
                        help='set to calculate square lattice model (default = false)')
    parser.add_argument('-Lx', metavar='Lx',dest='Lx', type=int, default=3,
                        help='set system size Lx  for square lattice (default = 3)')
    parser.add_argument('-Ly', metavar='Ly',dest='Ly', type=int, default=3,
                        help='set system size Ly  for square lattice (default = 3)')
    parser.add_argument('--periodic_y',dest='periodic_y', action="store_true",
                        help='set periodic boundary condision for y directioon (default = open)')
    return parser.parse_args()

if __name__ == "__main__":
    ## read params from command line
    args = parse_args()

    if args.calc_square:
        eig_val,eig_vec = Calc_GS_2d(args.m,args.Jz,args.Jxy,args.hx,args.hz,args.D,args.Lx,args.Ly,args.e_num,args.periodic,args.periodic_y)
        N = args.Lx * args.Ly
        print(f"{N}-site spin model on squre lattice")
        print(f"(Lx, Ly) = ({args.Lx}, {args.Ly})")
    else:
        eig_val,eig_vec = Calc_GS(args.m,args.Jz,args.Jxy,args.hx,args.hz,args.D,args.N,args.e_num,args.periodic)
        print(f"{args.N}-site spin chain")

    print(f"Ground state energy = {eig_val[0]}")
    for i in range(1,args.e_num):
        print(f"Excited states {i}: {eig_val[i]}")
