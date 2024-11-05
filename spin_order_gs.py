import torch
import groups.S2 as S2
import config as cfg
import numpy as np
from math import sqrt
from complex_num.complex_operation import *
import itertools
import context
import time
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse
import config as cfg
# from mpi4py import MPI
from mps.mps import *
from models import DQCP
from optim.ad_optim import optimize_state
#from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

def energy_2x2_2site(self,state):
        mps1 = state.sites[(0)]
        mps2 = state.sites[(1)]
        
        norm1 = torch.eye(size_complex(mps1)[2]**2,dtype=self.dtype,device=self.device)
        norm2 = torch.zeros((size_complex(mps1)[2]**2,size_complex(mps1)[2]**2),dtype=self.dtype,device=self.device)
        norm = torch.stack((norm1, norm2), dim=0)
        G_list_contract1 = einsum_complex('ijk,imn->jmkn',mps1,complex_conjugate(mps1))#.reshape(G_list[j].size()[1]**2,G_list[j].size()[2]**2))
        G_list_contract1 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), G_list_contract1)
        G_list_contract2 = einsum_complex('ijk,imn->jmkn',mps2,complex_conjugate(mps2))#.reshape(G_list[j].size()[1]**2,G_list[j].size()[2]**2))
        G_list_contract2 = view_complex((size_complex(mps2)[1]**2,size_complex(mps2)[2]**2), G_list_contract2)
        for j in range(0,self.N):
            if j % 2 == 0:
                norm = mm_complex(norm,G_list_contract1)
            else:
                norm = mm_complex(norm,G_list_contract2)
        norm = trace_complex(norm)
        
        theta = einsum_complex('ijk,lkm->ijlm',mps1,mps2)
        theta_O = einsum_complex('ijkl,kmln->imjn',(self.J*self.SSx+self.J*self.SSy+self.J*self.delta*self.SSz),theta)
        E1 = einsum_complex('ijkl,imkn->jmln',theta_O,complex_conjugate(theta))#.reshape(G_list[(j)%size].size()[1]**2,G_list[(j+1)%size].size()[2]**2)
        E1 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), E1)
        for i in range(1,self.N-1):
            if i % 2 == 1:
                E1 = mm_complex(E1,G_list_contract1)
            else:
                E1 = mm_complex(E1,G_list_contract2)
        # E1*=(self.N/2)
        theta = einsum_complex('ijk,lkm->ijlm',mps2,mps1)
        theta_O = einsum_complex('ijkl,kmln->imjn',(self.J*self.SSx+self.J*self.SSy+self.J*self.delta*self.SSz),theta)
        E2 = einsum_complex('ijkl,imkn->jmln',theta_O,complex_conjugate(theta))#.reshape(G_list[(j)%size].size()[1]**2,G_list[(j+1)%size].size()[2]**2)
        E2 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), E2)
        for i in range(1,self.N-1):
            if i % 2 == 0:
                E2 = mm_complex(E2,G_list_contract1)
            else:
                E2 = mm_complex(E2,G_list_contract2)
        # E2*=(self.N/2)
        E_nn = (trace_complex(E1)+trace_complex(E2))/norm/2

def lattice_to_site(coord):
        return (coord[0]%2)

J = 1.0
delta = 2.1
h = 270.27
N = 8

model = DQCP.DQCP(J=J, delta=delta, gx=3.7, bohr=5.78e-5, h=h, N=N)
state = read_ipeps('ex-mps_s8with32_chi10_j1_y_h0.9_state.json', vertexToSite=lattice_to_site)
energy_f=model.energy_2x2_2site_sz_gs
# print (energy_f(state).item())

################Left Canonical################
A2 = state.sites[(0)].detach().cpu().numpy()
A = A2[0] + 1j*A2[1]
sizeA = A.shape

B2 = state.sites[(1)].detach().cpu().numpy()
B = B2[0] + 1j*B2[1]
sizeB = B.shape

L0 = np.identity(sizeA[1])
L1 = np.identity(sizeB[1])
iter_max = 1000

def leftorthonormalize(A, B, L0, L1, tor=1.0e-8):
    sizeA = A.shape
    sizeB = B.shape
    L_old0 = L0/np.linalg.norm(L0, 'fro')
    L_old1 = L1/np.linalg.norm(L1, 'fro')
    AL0, L1 = np.linalg.qr(np.tensordot(L_old0, A, (1,1)).reshape(sizeA[0]*sizeA[1], sizeA[2]))
    AL1, L0 = np.linalg.qr(np.tensordot(L_old1, B, (1,1)).reshape(sizeB[0]*sizeB[1], sizeB[2]))
    lam = np.linalg.norm(L0, 'fro')
    L0 = L0/lam
    lam = np.linalg.norm(L1, 'fro')
    L1 = L1/lam
    #delta = np.linalg.norm(L-L_old, 'fro')
    for i in range(iter_max):
        #AL = np.transpose(AL.reshape(sizeA[1], sizeA[0], sizeA[2]), (1,0,2))
        #A_DL = np.transpose(np.tensordot(A, AL, (0,0)), (1,3,0,2)).reshape(sizeA[2]**2, sizeA[1]**2)
        #e, v = spr_linalg.eigs(A_DL, 1)
        #L = v.reshape(sizeA[1], sizeA[1])
        #temp, L = np.linalg.qr(L)
        #L = L/np.linalg.norm(L, 'fro')
        L_old0 = L0.copy()
        L_old1 = L1.copy()
        AL0, L1 = np.linalg.qr(np.tensordot(L0, A, (1,1)).reshape(sizeA[0]*sizeA[1], sizeA[2]))
        AL1, L0 = np.linalg.qr(np.tensordot(L1, B, (1,1)).reshape(sizeB[0]*sizeB[1], sizeB[2]))
        lam = np.linalg.norm(L0, 'fro')
        L0 = L0/lam
        lam = np.linalg.norm(L1, 'fro')
        L1 = L1/lam
        delta1 = abs(np.linalg.norm(L1-L_old1, 'fro')) + abs(np.linalg.norm(L0-L_old0, 'fro'))
        delta2 = abs(np.linalg.norm(L1+L_old1, 'fro')) + abs(np.linalg.norm(L0+L_old0, 'fro'))
        if delta1 < tor or delta2 < tor:
            break
        elif i==iter_max-1:
            print ("lconverge fail, delta=", delta1, delta2)
    return np.transpose(AL0.reshape(sizeA[1], sizeA[0], sizeA[2]), (1,0,2)), np.transpose(AL1.reshape(sizeB[1], sizeB[0], sizeB[2]), (1,0,2)), L0, L1

def rightorthonormalize(A, B, L0, L1, tor=1.0e-8):
    sizeA = A.shape
    A = np.transpose(A, (0,2,1))
    sizeB = B.shape
    B = np.transpose(B, (0,2,1))
    L_old0 = L0/np.linalg.norm(L0, 'fro')
    L_old1 = L1/np.linalg.norm(L1, 'fro')
    AL0, L1 = np.linalg.qr(np.tensordot(L_old0, A, (1,1)).reshape(sizeA[0]*sizeA[2], sizeA[1]))
    AL1, L0 = np.linalg.qr(np.tensordot(L_old1, B, (1,1)).reshape(sizeB[0]*sizeB[2], sizeB[1]))
    lam = np.linalg.norm(L0, 'fro')
    L0 = L0/lam
    lam = np.linalg.norm(L1, 'fro')
    L1 = L1/lam
    #delta = np.linalg.norm(L-L_old, 'fro')
    for i in range(iter_max):
        #AL = np.transpose(AL.reshape(sizeA[1], sizeA[0], sizeA[2]), (1,0,2))
        #A_DL = np.transpose(np.tensordot(A, AL, (0,0)), (1,3,0,2)).reshape(sizeA[1]**2, sizeA[2]**2)
        #e, v = spr_linalg.eigs(A_DL, 1)
        #L = v.reshape(sizeA[2], sizeA[2])
        #temp, L = np.linalg.qr(L)
        #L = L/np.linalg.norm(L, 'fro')
        L_old0 = L0.copy()
        L_old1 = L1.copy()
        AL0, L1 = np.linalg.qr(np.tensordot(L0, A, (1,1)).reshape(sizeA[0]*sizeA[2], sizeA[1]))
        AL1, L0 = np.linalg.qr(np.tensordot(L1, B, (1,1)).reshape(sizeB[0]*sizeB[2], sizeB[1]))
        lam = np.linalg.norm(L0, 'fro')
        L0 = L0/lam
        lam = np.linalg.norm(L1, 'fro')
        L1 = L1/lam
        delta1 = abs(np.linalg.norm(L1-L_old1, 'fro')) + abs(np.linalg.norm(L0-L_old0, 'fro'))
        delta2 = abs(np.linalg.norm(L1+L_old1, 'fro')) + abs(np.linalg.norm(L0+L_old0, 'fro'))
        if delta1 < tor or delta2 < tor:
            break
        elif i==iter_max-1:
            print ("rconverge fail, delta=", delta1, delta2)
    return np.transpose(AL0.reshape(sizeA[1], sizeA[0], sizeA[2]), (1,2,0)), np.transpose(AL1.reshape(sizeA[1], sizeA[0], sizeA[2]), (1,2,0)), np.transpose(L0,(1,0)), np.transpose(L1,(1,0))

AL, BL, L0, L1 = leftorthonormalize(A, B, L0, L1)
AR, BR, C0, C1 = rightorthonormalize(AL, BL, L0, L1)
u0, C0, v0 = linalg.svd(C0)
u1, C1, v1 = linalg.svd(C1)
AL = np.tensordot(np.conjugate(np.transpose(u1,(1,0))), AL, (1,1))
AL = np.transpose(np.tensordot(AL, u0, (2,0)),(1,0,2))
BL = np.tensordot(np.conjugate(np.transpose(u0,(1,0))), BL, (1,1))
BL = np.transpose(np.tensordot(BL, u1, (2,0)),(1,0,2))

Ar = torch.as_tensor(AL.real, dtype=torch.float64)
Ai = torch.as_tensor(AL.imag, dtype=torch.float64)
A1 = torch.stack((Ar, Ai), dim=0)
state.sites[(0)] = A1
Br = torch.as_tensor(BL.real, dtype=torch.float64)
Bi = torch.as_tensor(BL.imag, dtype=torch.float64)
A2 = torch.stack((Br, Bi), dim=0)
state.sites[(1)] = A2



order = model.energy_2x2_2site_sz_gs
print(f" order_z :{order(A1,A2):.15f}")
