import context
import time
import torch
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse
import config as cfg
from mps.mps import *
from models import DQCP
from complex_num.complex_operation import *
from optim.ad_optim import optimize_state
#from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)

torch.set_num_threads(1)
tStart = time.time()
torch.pi = torch.tensor(np.pi, dtype=torch.float64)
# Jx = 1.0
# Jz = 1.1
# Kx = 0.5
# Kz = 0.5
J = 1
delta = 2.1
h = 0
N = 8
numk = 0
k = (numk) * 2 * np.pi / N
#print (np.cos(k))
#print (np.sin(k))
numofstate = 30
unit_len = 2
# model = DQCP.DQCP(Jx, Jz, Kx, Kz, N)
model = DQCP.DQCP(J=J, delta=delta, gx=3.7, bohr=5.78e-5, h=h, N=N)
def lattice_to_site(coord):
    return (0)
state = read_ipeps("/home/23_takahashi/finite_tem/SrCo2V2O8/excitation_DQCP/ex-mps_n8_chi10_j1_y_h0_state.json", vertexToSite=lattice_to_site)
energy_f=model.energy_2x2_2site
#print (energy_f(state).item())

################Left Canonical################
A2 = state.sites[(0)].detach().cpu().numpy()
A = A2[0] + 1j*A2[1]
sizeA = A.shape
print(f"sizeA :{sizeA}")

B2 = state.sites[(1)].detach().cpu().numpy()
B = B2[0] + 1j*B2[1]
sizeB = B.shape
print(f"sizeB :{sizeB}")
##############make c tensor#############
# C1 = np.transpose(np.tensordot(A,B,(2,1)).reshape(2*sizeA[0],sizeA[1],sizeA[2]),(0,1,2))
C1 = np.transpose(np.tensordot(A,B,(2,1)),(0,2,1,3)).reshape(sizeA[0]**2,sizeA[1],sizeA[2])
########calculate norm##########

    # temp=contiguous_complex(einsum_complex('ijkl,ijmn->kmln', A, complex_conjugate(A)))

    # temp=temp.reshape(sizeA[1]**2,sizeA[2]**2)
# temp=np.tensordot(C1,C1.conj(),(0,0))
# for i in range(int(N/2)-1):
#     temp = np.tensordot(temp,C1,(1,1))
#     temp = np.transpose(np.tensordot(temp,C1.conj(),([2,3],[1,0])),(0,2,1,3))
# temp = np.transpose(temp,(0,2,1,3)).reshape(sizeA[1]**2,sizeA[2]**2)
# nor = temp.trace()
# print(f"nor : {nor}")
# print(nor.shape)
# C1 /= nor**(1/(N/2)/2)
# print(f"normvalue:{nor**(1/(N/2)/2)}")
# Cr = torch.as_tensor(C.real, dtype=torch.float64)
# Ci = torch.as_tensor(C.imag, dtype=torch.float64)
sizeC = C1.shape
##############rename sizeA##########
sizeA = sizeC
print(f"sizeC :{sizeC}") #######(4,10,10)

L0 = np.identity(sizeC[1])
# L1 = np.identity(sizeB[1])
iter_max = 1000

def leftorthonormalize(A, L0, tor=1.0e-8):
    sizeA = A.shape
    L_old = L0/np.linalg.norm(L0, 'fro')
    AL, L = np.linalg.qr(np.tensordot(L_old, A, (1,1)).reshape(sizeA[0]*sizeA[1], sizeA[2]))
    # print(f"AL {AL.shape}")
    lam = np.linalg.norm(L, 'fro')
    L = L/lam
    #delta = np.linalg.norm(L-L_old, 'fro')
    for i in range(iter_max):
        #AL = np.transpose(AL.reshape(sizeA[1], sizeA[0], sizeA[2]), (1,0,2))
        #A_DL = np.transpose(np.tensordot(A, AL, (0,0)), (1,3,0,2)).reshape(sizeA[2]**2, sizeA[1]**2)
        #e, v = spr_linalg.eigs(A_DL, 1)
        #L = v.reshape(sizeA[1], sizeA[1])
        #temp, L = np.linalg.qr(L)
        #L = L/np.linalg.norm(L, 'fro')
        L_old = L.copy()
        AL, L = np.linalg.qr(np.tensordot(L, A, (1,1)).reshape(sizeA[0]*sizeA[1], sizeA[2]))
        lam = np.linalg.norm(L, 'fro')
        L = L/lam
        delta1 = np.linalg.norm(L-L_old, 'fro')
        delta2 = np.linalg.norm(L+L_old, 'fro')
        if delta1 < tor or delta2 < tor:
            break
        elif i==iter_max-1:
            print ("lconverge fail, delta=", delta)
    return np.transpose(AL.reshape(sizeA[1], sizeA[0], sizeA[2]), (1,0,2)), L, lam


def rightorthonormalize(A, L0, tor=1.0e-8):
    sizeA = A.shape
    A = np.transpose(A, (0,2,1))
    L_old = L0/np.linalg.norm(L0, 'fro')
    AL, L = np.linalg.qr(np.tensordot(L_old, A, (1,1)).reshape(sizeA[0]*sizeA[2], sizeA[1]))
    lam = np.linalg.norm(L, 'fro')
    L = L/lam
    #delta = np.linalg.norm(L-L_old, 'fro')
    for i in range(iter_max):
        #AL = np.transpose(AL.reshape(sizeA[1], sizeA[0], sizeA[2]), (1,0,2))
        #A_DL = np.transpose(np.tensordot(A, AL, (0,0)), (1,3,0,2)).reshape(sizeA[1]**2, sizeA[2]**2)
        #e, v = spr_linalg.eigs(A_DL, 1)
        #L = v.reshape(sizeA[2], sizeA[2])
        #temp, L = np.linalg.qr(L)
        #L = L/np.linalg.norm(L, 'fro')
        L_old = L.copy()
        AL, L = np.linalg.qr(np.tensordot(L, A, (1,1)).reshape(sizeA[0]*sizeA[2], sizeA[1]))
        lam = np.linalg.norm(L, 'fro')
        L = L/lam
        delta1 = np.linalg.norm(L-L_old, 'fro')
        delta2 = np.linalg.norm(L+L_old, 'fro')
        if delta1 < tor or delta2 < tor:
            break
        elif i==iter_max-1:
            print ("rconverge fail, delta=", delta)
        # print(f"ARf {np.transpose(AL.reshape(sizeA[1], sizeA[0], sizeA[2]), (1,2,0)).shape}")
    return np.transpose(AL.reshape(sizeA[1], sizeA[0], sizeA[2]), (1,2,0)), np.transpose(L,(1,0)), lam

AL, L, lam = leftorthonormalize(C1, L0)
AR, C, lam = rightorthonormalize(AL, L0)
u, C, v = linalg.svd(C)
AL = np.tensordot(np.conjugate(np.transpose(u,(1,0))), AL, (1,1))
AL = np.transpose(np.tensordot(AL, u, (2,0)),(1,0,2))
Ar = torch.as_tensor(AL.real, dtype=torch.float64)
Ai = torch.as_tensor(AL.imag, dtype=torch.float64)
A = torch.stack((Ar, Ai), dim=0)
########reshape A (2,4,10,10) -> (2,2,2,10,10)
# A = A.reshape(2,int(sizeA[0]/2),int(sizeA[0]/2),sizeA[1],sizeA[2])
# print(f"Ashape : {A.shape}")
state.sites[(0)] = A
# print (f"new :{energy_f(state).item()}")
A_DL = np.transpose(np.tensordot(AL, np.conjugate(AL), (0,0)), (1,3,0,2)).reshape(sizeA[1]**2, sizeA[2]**2)
#print (spr_linalg.eigs(A_DL, 1, which='LM'))
#print (C)
lamr = torch.as_tensor(np.diag(1/C).real, dtype=torch.float64)
lami = torch.zeros((lamr.size()[0],lamr.size()[1]), dtype=torch.float64)
lam_inv = torch.stack((lamr, lami), dim=0)
#print (sizeA[0]*sizeA[1]*sizeA[2])
################MPO################
# MPO1 = torch.zeros((2,2,2,5,5), dtype=torch.float64)
# MPON = torch.zeros((2,2,2,5,5), dtype=torch.float64)
# MPOi = torch.zeros((2,2,2,5,5), dtype=torch.float64)
# MPO1[0][0][0][0][4] = MPO1[0][1][1][0][4] = 1.0
# np.set_printoptions(threshold=np.inf)
# # print(f"MPO1 : {MPO1}")
# MPO1[0][0][1][0][1] = J /2.0
# MPO1[0][1][0][0][1] = J /2.0
# # MPO1[0][1][2][0][1] = MPO1[0][1][2][5][5] = J / np.sqrt(2.)
# # MPO1[0][2][1][0][1] = MPO1[0][2][1][5][5] = J / np.sqrt(2.)
# MPO1[1][0][1][0][2] = J /2.0
# MPO1[1][1][0][0][2] = -J /2.0
# # MPO1[1][1][2][0][2] = MPO1[1][1][2][6][6] = J / np.sqrt(2.)
# # MPO1[1][2][1][0][2] = MPO1[1][2][1][6][6] = -J / np.sqrt(2.)
# MPO1[0][0][0][0][3] = J * delta / 2.0
# MPO1[0][1][1][0][3] = -J * delta/ 2.0
# # MPO1[0][2][2][0][3] = MPO1[0][2][2][7][7] = -J
# # print(f"MPO1 :{MPO1}")

# MPON[0][0][0][0][0] = MPON[0][1][1][0][0] = 1.0
# MPON[0][0][1][1][0] = 1.0/2.0
# MPON[0][1][0][1][0] = 1.0/2.0
# # MPON[0][1][2][1][0] = MPON[0][1][2][5][5] = 1.0/np.sqrt(2.)
# # MPON[0][2][1][1][0] = MPON[0][2][1][5][5] = 1.0/np.sqrt(2.)
# MPON[1][0][1][2][0] = 1.0/2.0
# MPON[1][1][0][2][0] = -1.0/2.0
# # MPON[1][1][2][2][0] = MPON[1][1][2][6][6] = 1.0/np.sqrt(2.)
# # MPON[1][2][1][2][0] = MPON[1][2][1][6][6] = -1.0/np.sqrt(2.)
# MPON[0][0][0][3][0] = 1.0 / 2.0
# MPON[0][1][1][3][0] = -1.0 / 2.0
# # MPON[0][2][2][3][0] = MPON[0][2][2][7][7] = -1.0

# MPOi[0][0][0][0][0] = MPOi[0][1][1][0][0] = 1.0
# MPOi[0][0][0][4][4] = MPOi[0][1][1][4][4] = 1.0
# MPOi[0][0][1][1][0] = 1.0/2.0
# MPOi[0][1][0][1][0] = 1.0/2.0
# # MPOi[0][1][2][1][0] = 1.0/np.sqrt(2.)
# # MPOi[0][2][1][1][0] = 1.0/np.sqrt(2.)
# MPOi[1][0][1][2][0] = 1.0/2.0
# MPOi[1][1][0][2][0] = -1.0/2.0
# # MPOi[1][1][2][2][0] = 1.0/np.sqrt(2.)
# # MPOi[1][2][1][2][0] = -1.0/np.sqrt(2.)
# MPOi[0][0][0][3][0] = 1.0 / 2.0
# MPOi[0][1][1][3][0] = -1.0 / 2.0
# # MPOi[0][2][2][3][0] = -1.0
# MPOi[0][0][1][4][1] = J / 2.0
# MPOi[0][1][0][4][1] = J / 2.0
# # MPOi[0][1][2][4][1] = J / np.sqrt(2.)
# # MPOi[0][2][1][4][1] = J / np.sqrt(2.)
# MPOi[1][0][1][4][2] = J / 2.0
# MPOi[1][1][0][4][2] = -J / 2.0
# # MPOi[1][1][2][4][2] = J / np.sqrt(2.)
# # MPOi[1][2][1][4][2] = -J / np.sqrt(2.)
# MPOi[0][0][0][4][3] = J * delta / 2.0
# MPOi[0][1][1][4][3] = -J * delta / 2.0

################MPO################
# MPO1 = torch.zeros((2,2,2,5,5), dtype=torch.float64)
MPON = torch.zeros((2,2,2,5,5), dtype=torch.float64)
MPOi = torch.zeros((2,2,2,5,5), dtype=torch.float64)
# MPO1[0][0][0][0][4] = MPO1[0][1][1][0][4] = 1.0
# np.set_printoptions(threshold=np.inf)
# # print(f"MPO1 : {MPO1}")
# MPO1[0][0][1][0][1] = J /2.0
# MPO1[0][1][0][0][1] = J /2.0
# # MPO1[0][1][2][0][1] = MPO1[0][1][2][5][5] = J / np.sqrt(2.)
# # MPO1[0][2][1][0][1] = MPO1[0][2][1][5][5] = J / np.sqrt(2.)
# MPO1[1][0][1][0][2] = J /2.0
# MPO1[1][1][0][0][2] = -J /2.0
# # MPO1[1][1][2][0][2] = MPO1[1][1][2][6][6] = J / np.sqrt(2.)
# # MPO1[1][2][1][0][2] = MPO1[1][2][1][6][6] = -J / np.sqrt(2.)
# MPO1[0][0][0][0][3] = J * delta / 2.0
# MPO1[0][1][1][0][3] = -J * delta/ 2.0
# MPO1[0][2][2][0][3] = MPO1[0][2][2][7][7] = -J
# print(f"MPO1 :{MPO1}")

MPON[0][0][0][0][4] = MPON[0][1][1][0][4] = 1.0
# MPON[0][0][0][4][4] = MPON[0][1][1][4][4] = 1.0
# MPON[0][0][1][0][1] = J/2.0
MPON[0][1][0][0][1] = J/2.0
MPON[0][0][1][1][4] = 1.0
# MPON[0][1][0][1][4] = 1.0/2.0
# MPON[0][1][2][1][0] = MPON[0][1][2][5][5] = 1.0/np.sqrt(2.)
# MPON[0][2][1][1][0] = MPON[0][2][1][5][5] = 1.0/np.sqrt(2.)
MPON[0][0][1][0][2] = J/2.0
# MPON[1][1][0][0][2] = -J/2.0
# MPON[1][0][1][2][4] = 1.0/2.0
MPON[0][1][0][2][4] = 1.0
# MPON[1][1][2][2][0] = MPON[1][1][2][6][6] = 1.0/np.sqrt(2.)
# MPON[1][2][1][2][0] = MPON[1][2][1][6][6] = -1.0/np.sqrt(2.)
MPON[0][0][0][0][3] = J*delta / 2.0
MPON[0][1][1][0][3] = -J*delta / 2.0
MPON[0][0][0][3][4] = 1.0 / 2.0
MPON[0][1][1][3][4] = -1.0 / 2.0
# MPON[0][2][2][3][0] = MPON[0][2][2][7][7] = -1.0


MPOi[0][0][0][0][0] = MPOi[0][1][1][0][0] = 1.0
MPOi[0][0][0][4][4] = MPOi[0][1][1][4][4] = 1.0
MPOi[0][0][1][1][0] = 1.0
# MPOi[0][1][0][1][0] = 1.0/2.0
# MPOi[0][1][2][1][0] = 1.0/np.sqrt(2.)
# MPOi[0][2][1][1][0] = 1.0/np.sqrt(2.)
# MPOi[1][0][1][2][0] = 1.0/2.0
MPOi[0][1][0][2][0] = 1.0
# MPOi[1][1][2][2][0] = 1.0/np.sqrt(2.)
# MPOi[1][2][1][2][0] = -1.0/np.sqrt(2.)
MPOi[0][0][0][3][0] = 1.0 / 2.0

MPOi[0][1][1][3][0] = -1.0 / 2.0
# MPOi[0][2][2][3][0] = -1.0
# MPOi[0][0][1][4][1] = J / 2.0
MPOi[0][1][0][4][1] = J / 2.0
# MPOi[0][1][2][4][1] = J / np.sqrt(2.)
# MPOi[0][2][1][4][1] = J / np.sqrt(2.)
MPOi[0][0][1][4][2] = J / 2.0
# MPOi[1][1][0][4][2] = -J / 2.0
# MPOi[1][1][2][4][2] = J / np.sqrt(2.)
# MPOi[1][2][1][4][2] = -J / np.sqrt(2.)
MPOi[0][0][0][4][3] = J * delta / 2.0
MPOi[0][1][1][4][3] = -J * delta / 2.0

###########truncation of mpo###########
mpo_list = []
for i in range(int(N/2)):
    # if i==0:
    #     # mpo1 = contiguous_complex(einsum_complex('ijkl,mnlp->ijmnkp', MPO1, MPOi))
    #     mpo1 = contiguous_complex(einsum_complex('ijkl,mnlp->imjnkp', MPO1, MPOi))
    #     mpo1 = view_complex((sizeA[0],sizeA[0],5,5),mpo1)
    #     mpo_list.append(mpo1)
        # mpo_list.append(einsum_complex('ijkl,mnlp->ijmnkp', MPO1, MPOi))
    if i==int(N/2-1):
        # mpo_list.append(einsum_complex('ijkl,mnlp->ijmnkp', MPON, MPO1).reshape(2,sizeA[0],sizeA[0],5,5))
        mpoN = contiguous_complex(einsum_complex('ijkl,mnlp->imjnkp', MPOi, MPON))
        mpoN = view_complex((sizeA[0],sizeA[0],5,5),mpoN)
        mpo_list.append(mpoN)
        # mpo_list.append(einsum_complex('ijkl,mnlp->ijmnkp', MPON, MPO1))
    else:
        # mpo_list.append(einsum_complex('ijkl,mnlp->ijmnkp', MPOi, MPOi).reshape(2,sizeA[0],sizeA[0],5,5))
        mpoi = contiguous_complex(einsum_complex('ijkl,mnlp->imjnkp', MPOi, MPOi))
        mpoi = view_complex((sizeA[0],sizeA[0],5,5),mpoi)
        mpo_list.append(mpoi)
        # mpo_list.append(einsum_complex('ijkl,mnlp->ijmnkp', MPOi, MPOi))
print(f"mposize :{mpo_list[0].shape}")
# print(f"mposize :{mpo_list[0]==mpo_list[1]}")

# A = A.reshape()
print(f"sizeA:{A.shape}")
for i in range(int(N/2)):
    # temp=contiguous_complex(einsum_complex('ijkl,ijmn->kmln', A, complex_conjugate(A)))
    temp=contiguous_complex(einsum_complex('ijk,imn->jmkn', A, complex_conjugate(A)))
    temp=view_complex((sizeA[1]**2,sizeA[2]**2),temp)
    if i==0:
        nor=temp.clone()
    elif i==N/2-1:
        # sum of diagonal, maybe it is summention of eigenvalue
        nor=torch.trace(mm_complex(nor, temp)[0])
    else:
        nor=mm_complex(nor, temp)
    print(f"nor : {nor}")
    print(nor.shape)
# A /= nor**(1/(N/2)/2)
# print(f"mposize:{mpo_list[0]}")
# print(f"MPO1size : {MPO1.shape}")
# temp1 = einsum_complex('ijkl,imjnop->mnkolp', A, mpo_list[0])
temp1 = einsum_complex('ijk,ilmn->jmknl', A, mpo_list[0])
# print(f"temp1size : {temp1.shape}")
# temp1 = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', temp1, complex_conjugate(A)))
temp1 = contiguous_complex(einsum_complex('jmkni,ilo->jmlkno', temp1, complex_conjugate(A)))
# print(f"temp1size : {temp1.shape}")
temp1 = view_complex((5*sizeA[1]**2,5*sizeA[2]**2),temp1)
# tempN = einsum_complex('ijkl,imjnop->mnkolp', A, mpo_list[int(N/2-1)])
tempN = einsum_complex('ijk,ilmn->jmknl', A, mpo_list[-1])
# tempN = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', tempN, complex_conjugate(A)))
tempN = contiguous_complex(einsum_complex('jmkni,ilo->jmlkno', tempN, complex_conjugate(A)))
tempN = view_complex((5*sizeA[1]**2,5*sizeA[2]**2),tempN)
# tempi = einsum_complex('ijkl,imjnop->mnkolp', A, mpo_list[1])
tempi = einsum_complex('ijk,ilmn->jmknl', A, mpo_list[1])
# tempi = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', tempi, complex_conjugate(A)))
tempi = contiguous_complex(einsum_complex('jmkni,ilo->jmlkno', tempi, complex_conjugate(A)))
tempi = view_complex((5*sizeA[1]**2,5*sizeA[2]**2),tempi)
temp2 = mm_complex(tempi, tempi)
ene = mm_complex(temp1, temp2)
for i in range(int(N/2/2-2)):
    ene = mm_complex(ene, temp2)
ene = torch.trace(mm_complex(ene, tempN)[0])
print(f"ene:{ene.shape}")
print(f"ene:{(ene/N/nor).item()}")
print(f"size {sizeA}")
print(ene.item())

################Excited state################
B_grad = torch.zeros((sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64).requires_grad_(True)
B = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
B[0] = B_grad
Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
N_cv = torch.zeros((2,sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
N_cm = torch.zeros((2,sizeA[0]*sizeA[1]*sizeA[2],sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
for i in range(int(N/2/unit_len)):
    if i==int(N/2/unit_len)-1:
        for j in range(unit_len):
            Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
            if j==unit_len-1:
                func_up = A + einsum_complex('ijk,kn->ijn',B,lam_inv)
                temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
                DL = einsum_complex('ijkl,klmn->ijmn',temp,DL)
                N_cv = contiguous_complex(einsum_complex('ijk,kljn->inl',func_up,DL))
                N_cv = view_complex((sizeA[0]*sizeA[1]*sizeA[2]), N_cv)
            elif j==0:
                Bk[0] = torch.tensor(np.cos(-k*(j+1))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+1))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                temp_down = einsum_complex('im,jmn->jin',lam_inv,complex_conjugate(A))
            else:
                Bk[0] = torch.tensor(np.cos(-k*(j+1))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+1))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                func_down = complex_conjugate(A)
                temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
                temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))    
    elif i==0:
        for j in range(unit_len):
            Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
            if j==0:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                # print(f"Bk {Bk}")
                temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                temp_down = complex_conjugate(A)
            else:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                func_down = complex_conjugate(A)
                temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
                temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
        DL = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
    else:
        for j in range(unit_len):
            Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
            if j==0:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                temp_down = complex_conjugate(A)
            else:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                func_down = complex_conjugate(A)
                temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
                temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
        temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
        DL = einsum_complex('ijkl,klmn->ijmn',DL,temp)
        # print(f"DL : {DL.shape}")

print (f"N_cv:{size_complex(N_cv)}")
print(N_cv[0][0])
dev_accu = torch.zeros((sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
for i in range(sizeA[0]*sizeA[1]*sizeA[2]):
    N_cv[0][i].backward(torch.ones_like(N_cv[0][i]), retain_graph=True)
    temp = B_grad.grad.view(sizeA[0]*sizeA[1]*sizeA[2]).clone()
    devr = temp.clone() - dev_accu
    dev_accu = temp.clone()
    N_cv[1][i].backward(torch.ones_like(N_cv[1][i]), retain_graph=True)
    temp = B_grad.grad.view(sizeA[0]*sizeA[1]*sizeA[2]).clone()
    devi = temp.clone() - dev_accu
    dev_accu = temp.clone()
    N_cm[0][i] = devr
    N_cm[1][i] = devi
print (f"N_cm :{size_complex(N_cm)}")
#N_cm+= transpose_complex(complex_conjugate(N_cm.clone()))
#N_cm = N_cm.clone()/2.0
print (f" item {N_cm[0][0][1].item(),N_cm[0][1][0].item(),N_cm[1][0][1].item(),N_cm[1][1][0].item(),N_cm[0][0][0].item(),N_cm[1][0][0].item()}")
N_cm2 = N_cm.detach().cpu().numpy()
N_cmn = N_cm2[0] + 1j*N_cm2[1]
N_cmn = N_cmn/np.linalg.norm(N_cmn, 'fro')
e, v = np.linalg.eig(N_cmn)
idx = np.argsort(e.real)   
e = e[idx]
v = v[:,idx]
print (e)
num = 0
for i in range(sizeA[0]*sizeA[1]*sizeA[2]):
    if e[i].real < 0.00001:
        num+=1
print (num)

############effective Hamiltonian########

##########make hamiltonian#########
X = torch.tensor([[0,1],[1,0]])
model = DQCP.DQCP(J, delta=delta, gx=3.7, bohr=5.78e-5, h=h, N=N)
SSx, SSy, SSz, SX, SY, SZ, SS = model.get_h()
ham_1 = J*SSx+J*SSy+J*delta*SSz
print(SY)


B_grad = torch.zeros((sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64).requires_grad_(True)
B = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
B[0] = B_grad
H_cv = torch.zeros((2,sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
H_cm = torch.zeros((2,sizeA[0]*sizeA[1]*sizeA[2],sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
for i in range(int(N/2/unit_len)):
    if i==int(N/2/unit_len)-1:
        for j in range(unit_len):
            Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
            if j==unit_len-1:
                func_up = A + einsum_complex('ijk,kn->ijn',B,lam_inv)
                func_up = einsum_complex('ijk,ilmn->jmknl',func_up,MPO1)
                temp = einsum_complex('ijkmn,iab->jkamnb',temp_up,temp_down)
                DL = einsum_complex('ijmkln,klnabc->ijmabc',temp,DL)
                H_cv = contiguous_complex(einsum_complex('ijmnk,mnlija->kal',func_up,DL))
                H_cv = view_complex((sizeA[0]*sizeA[1]*sizeA[2]), H_cv)
            elif j==0:
                Bk[0] = torch.tensor(np.cos(-k*(j+1))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+1))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                temp_up = einsum_complex('ijk,ilmn->ljmkn',temp_up,MPOi)
                temp_down = einsum_complex('im,jmn->jin',lam_inv,complex_conjugate(A))
            else:
                Bk[0] = torch.tensor(np.cos(-k*(j+1))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+1))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                func_up = einsum_complex('ijk,ilmn->ljmkn',func_up,MPOi)
                func_down = complex_conjugate(A)
                temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],8,sizeA[2],8),einsum_complex('ijkmn,lmnab->iljkab',temp_up,func_up))
                temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
    elif i==0:
        for j in range(unit_len):
            Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
            if j==0:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                temp_up = temp_up.reshape(2,int(sizeA[0]/2),int(sizeA[0]/2),sizeA[1],sizeA[2])
                temp_up = einsum_complex('ijkl,ijmn->mnkl',temp_up,ham_1)
                print(f"temp:{temp_up.shape}")
                temp_down = complex_conjugate(A)
            else:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                func_up = einsum_complex('ijkl,ijmn->klmn',func_up,MPO)
                func_down = complex_conjugate(A)
                temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],8,sizeA[2],8),einsum_complex('ijkmn,lmnab->iljkab',temp_up,func_up))
                temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
        DL = einsum_complex('ijkmn,iab->jkamnb',temp_up,temp_down)
    elif i==int(N/unit_len)-2:
        for j in range(unit_len):
            Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
            if j==0:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                temp_up = einsum_complex('ijk,ilmn->ljmkn',temp_up,MPOi)
                temp_down = complex_conjugate(A)
            elif j==unit_len-1:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                func_up = einsum_complex('ijk,ilmn->ljmkn',func_up,MPON)
                func_down = complex_conjugate(A)
                temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],8,sizeA[2],8),einsum_complex('ijkmn,lmnab->iljkab',temp_up,func_up))
                temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
            else:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                func_up = einsum_complex('ijk,ilmn->ljmkn',func_up,MPOi)
                func_down = complex_conjugate(A)
                temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],8,sizeA[2],8),einsum_complex('ijkmn,lmnab->iljkab',temp_up,func_up))
                temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
        temp = einsum_complex('ijkmn,iab->jkamnb',temp_up,temp_down)
        DL = einsum_complex('ijmkln,klnabc->ijmabc',DL,temp)
    else:
        for j in range(unit_len):
            Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
            if j==0:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                temp_up = einsum_complex('ijk,ilmn->ljmkn',temp_up,MPOi)
                temp_down = complex_conjugate(A)
            else:
                Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
                Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
                func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
                func_up = einsum_complex('ijk,ilmn->ljmkn',func_up,MPOi)
                func_down = complex_conjugate(A)
                temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],8,sizeA[2],8),einsum_complex('ijkmn,lmnab->iljkab',temp_up,func_up))
                temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
        temp = einsum_complex('ijkmn,iab->jkamnb',temp_up,temp_down)
        DL = einsum_complex('ijmkln,klnabc->ijmabc',DL,temp)

print (f"H_size :{size_complex(H_cv)}")
dev_accu = torch.zeros((sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
for i in range(sizeA[0]*sizeA[1]*sizeA[2]):
    H_cv[0][i].backward(torch.ones_like(H_cv[0][i]), retain_graph=True)
    temp = B_grad.grad.view(sizeA[0]*sizeA[1]*sizeA[2]).clone()
    devr = temp.clone() - dev_accu
    dev_accu = temp.clone()
    H_cv[1][i].backward(torch.ones_like(H_cv[1][i]), retain_graph=True)
    temp = B_grad.grad.view(sizeA[0]*sizeA[1]*sizeA[2]).clone()
    devi = temp.clone() - dev_accu
    dev_accu = temp.clone()
    H_cm[0][i] = devr
    H_cm[1][i] = devi
print (size_complex(H_cm))
#H_cm+= transpose_complex(complex_conjugate(H_cm.clone()))
#H_cm = H_cm.clone()/2.0
print (f"H_cm : {H_cm[0][0][1],H_cm[0][1][0],H_cm[1][0][1],H_cm[1][1][0]}")
#H_cm2 = H_cm.detach().cpu().numpy()
#H_cmn = H_cm2[0] + 1j*H_cm2[1]
#e, v = np.linalg.eig(H_cmn)
#idx = np.argsort(e.real)   
#e = e[idx]
#v = v[:,idx]
#print (e)
#for i in range(200):
#    for j in range(200):
#        if 0.00001 < abs(H_cm[0][i][j]-H_cm[0][j][i]) or 0.00001 < abs(H_cm[1][i][j]+H_cm[1][j][i]) or 0.00001 < abs(H_cm[1][i][i]):
#            print ('error',i,j,H_cm[0][i][j].item(), H_cm[0][j][i].item(), H_cm[1][i][j].item(), H_cm[1][j][i].item(), H_cm[0][i][i].item(), H_cm[1][i][i].item(),
#                   abs(H_cm[0][i][j]-H_cm[0][j][i]), abs(H_cm[1][i][j]+H_cm[1][j][i]), abs(H_cm[1][i][i]))
                   

N_cm2 = N_cm.detach().cpu().numpy()
N_cmn = N_cm2[0] + 1j*N_cm2[1]
N_cmn = N_cmn/np.linalg.norm(N_cmn, 'fro')
N_cmn_inv = linalg.pinvh(N_cmn)#, rcond=0.0)
#N_cmr = N_cm2[0]
#N_cmi = N_cm2[1]
#N_cmr_inv = linalg.inv(N_cmr)
#N_cmi_inv = linalg.inv(N_cmi)
#inv_r = linalg.inv(N_cmr + np.matmul(np.matmul(N_cmi,N_cmr_inv),N_cmi))
#inv_i = -linalg.inv(N_cmi + np.matmul(np.matmul(N_cmr,N_cmi_inv),N_cmr))
#N_cmn_inv = inv_r + 1j*inv_i
H_cm2 = H_cm.detach().cpu().numpy()
H_cmn = H_cm2[0] + 1j*H_cm2[1]
H_cmn = H_cmn/np.linalg.norm(H_cmn, 'fro')
print (H_cmn[0][1].real, H_cmn[1][0].real, H_cmn[0][1].imag, H_cmn[1][0].imag)
#for i in range(200):
#    for j in range(200):
#        if 0.00001 < abs(H_cmn[i][j].real-H_cmn[j][i].real) or 0.00001 < abs(H_cmn[i][j].imag+H_cmn[j][i].imag) or 0.00001 < abs(H_cmn[i][i].imag):
#            print ('error',H_cmn[i][j].real, H_cmn[j][i].real, H_cmn[i][j].imag, H_cmn[j][i].imag, H_cmn[i][i].real, H_cmn[i][i].imag)
print (N_cmn[0][1].real, N_cmn[1][0].real, N_cmn[0][1].imag, N_cmn[1][0].imag)
#for i in range(200):
#    for j in range(200):
#        if 0.00001 < abs(N_cmn[i][j].real-N_cmn[j][i].real) or 0.00001 < abs(N_cmn[i][j].imag+N_cmn[j][i].imag) or 0.00001 < abs(N_cmn[i][i].imag):
#            print ('error',N_cmn[i][j].real, N_cmn[j][i].real, N_cmn[i][j].imag, N_cmn[j][i].imag, N_cmn[i][i].real, N_cmn[i][i].imag)
print (N_cmn_inv[0][1].real, N_cmn_inv[1][0].real, N_cmn_inv[0][1].imag, N_cmn_inv[1][0].imag)
#for i in range(200):
#    for j in range(200):
#        if 0.00001 < abs(N_cmn_inv[i][j].real-N_cmn_inv[j][i].real) or 0.00001 < abs(N_cmn_inv[i][j].imag+N_cmn_inv[j][i].imag) or 0.00001 < abs(N_cmn_inv[i][i].imag):
#            print ('error',N_cmn_inv[i][j].real, N_cmn_inv[j][i].real, N_cmn_inv[i][j].imag, N_cmn_inv[j][i].imag, N_cmn_inv[i][i].real, N_cmn_inv[i][i].imag)
#for i in range(200):
#    for j in range(200):
#        if 0.001 < abs(np.matmul(N_cmn_inv,H_cmn)[i][j].real-np.matmul(N_cmn_inv,H_cmn)[j][i].real) and 0.001 < abs(np.matmul(N_cmn_inv,H_cmn)[i][j].imag+np.matmul(N_cmn_inv,H_cmn)[j][i].imag):
#            print ('error',np.matmul(N_cmn_inv,H_cmn)[i][j].real, np.matmul(N_cmn_inv,H_cmn)[j][i].real, np.matmul(N_cmn_inv,H_cmn)[i][j].imag, np.matmul(N_cmn_inv,H_cmn)[j][i].imag)
e, v = np.linalg.eig(np.matmul(N_cmn_inv,H_cmn))#, numofstate, sigma=-1.0, which='SM')
idx = np.argsort(e)   
e = e[idx]
v = v[:,idx]
#for i in range(e.shape[0]):
#    if (abs(e[i].imag)<0.001):
#        print (e[i].real)
print (e)


# B_grad = torch.zeros((sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64).requires_grad_(True)
# B = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
# B[0] = B_grad
# Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
# N_cv = torch.zeros((2,sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
# N_cm = torch.zeros((2,sizeA[0]*sizeA[1]*sizeA[2],sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
# for i in range(int(N/unit_len)):
#     if i==int(N/unit_len)-1:
#         for j in range(unit_len):
#             Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             if j==unit_len-1:
#                 func_up = A + einsum_complex('ijk,kn->ijn',B,lam_inv)
#                 temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
#                 DL = einsum_complex('ijkl,klmn->ijmn',temp,DL)
#                 N_cv = contiguous_complex(einsum_complex('ijk,kljn->inl',func_up,DL))
#                 N_cv = view_complex((sizeA[0]*sizeA[1]*sizeA[2]), N_cv)
#             elif j==0:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+1))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+1))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 temp_down = einsum_complex('im,jmn->jin',lam_inv,complex_conjugate(A))
#             else:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+1))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+1))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 func_down = complex_conjugate(A)
#                 temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                 temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))    
#     elif i==0:
#         for j in range(unit_len):
#             Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             if j==0:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 # print(f"Bk {Bk}")
#                 temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 temp_down = complex_conjugate(A)
#             else:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 func_down = complex_conjugate(A)
#                 temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                 temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#         DL = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
#     else:
#         for j in range(unit_len):
#             Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             if j==0:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 temp_down = complex_conjugate(A)
#             else:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 func_down = complex_conjugate(A)
#                 temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                 temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#         temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
#         DL = einsum_complex('ijkl,klmn->ijmn',DL,temp)
#         # print(f"DL : {DL.shape}")
            
# print (f"N_cv:{size_complex(N_cv)}")
# dev_accu = torch.zeros((sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
# for i in range(sizeA[0]*sizeA[1]*sizeA[2]):
#     N_cv[0][i].backward(torch.ones_like(N_cv[0][i]), retain_graph=True)
#     temp = B_grad.grad.view(sizeA[0]*sizeA[1]*sizeA[2]).clone()
#     devr = temp.clone() - dev_accu
#     dev_accu = temp.clone()
#     N_cv[1][i].backward(torch.ones_like(N_cv[1][i]), retain_graph=True)
#     temp = B_grad.grad.view(sizeA[0]*sizeA[1]*sizeA[2]).clone()
#     devi = temp.clone() - dev_accu
#     dev_accu = temp.clone()
#     N_cm[0][i] = devr
#     N_cm[1][i] = devi
# print (f"N_cm :{size_complex(N_cm)}")
# #N_cm+= transpose_complex(complex_conjugate(N_cm.clone()))
# #N_cm = N_cm.clone()/2.0
# print (f" item {N_cm[0][0][1].item(),N_cm[0][1][0].item(),N_cm[1][0][1].item(),N_cm[1][1][0].item(),N_cm[0][0][0].item(),N_cm[1][0][0].item()}")
# N_cm2 = N_cm.detach().cpu().numpy()
# N_cmn = N_cm2[0] + 1j*N_cm2[1]
# N_cmn = N_cmn/np.linalg.norm(N_cmn, 'fro')
# e, v = np.linalg.eig(N_cmn)
# idx = np.argsort(e.real)   
# e = e[idx]
# v = v[:,idx]
# print (e)
# num = 0
# for i in range(sizeA[0]*sizeA[1]*sizeA[2]):
#     if e[i].real < 0.00001:
#         num+=1
# print (num)

# B_grad = torch.zeros((sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64).requires_grad_(True)
# B = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
# B[0] = B_grad
# H_cv = torch.zeros((2,sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
# H_cm = torch.zeros((2,sizeA[0]*sizeA[1]*sizeA[2],sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
# for i in range(int(N/unit_len)):
#     if i==int(N/unit_len)-1:
#         for j in range(unit_len):
#             Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             if j==unit_len-1:
#                 func_up = A + einsum_complex('ijk,kn->ijn',B,lam_inv)
#                 func_up = einsum_complex('ijk,ilmn->jmknl',func_up,MPO1)
#                 temp = einsum_complex('ijkmn,iab->jkamnb',temp_up,temp_down)
#                 DL = einsum_complex('ijmkln,klnabc->ijmabc',temp,DL)
#                 H_cv = contiguous_complex(einsum_complex('ijmnk,mnlija->kal',func_up,DL))
#                 H_cv = view_complex((sizeA[0]*sizeA[1]*sizeA[2]), H_cv)
#             elif j==0:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+1))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+1))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 temp_up = einsum_complex('ijk,ilmn->ljmkn',temp_up,MPOi)
#                 temp_down = einsum_complex('im,jmn->jin',lam_inv,complex_conjugate(A))
#             else:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+1))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+1))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 func_up = einsum_complex('ijk,ilmn->ljmkn',func_up,MPOi)
#                 func_down = complex_conjugate(A)
#                 temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],8,sizeA[2],8),einsum_complex('ijkmn,lmnab->iljkab',temp_up,func_up))
#                 temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#     elif i==0:
#         for j in range(unit_len):
#             Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             if j==0:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 temp_up = einsum_complex('ijk,ilmn->ljmkn',temp_up,MPOi)
#                 temp_down = complex_conjugate(A)
#             else:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 func_up = einsum_complex('ijk,ilmn->ljmkn',func_up,MPOi)
#                 func_down = complex_conjugate(A)
#                 temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],8,sizeA[2],8),einsum_complex('ijkmn,lmnab->iljkab',temp_up,func_up))
#                 temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#         DL = einsum_complex('ijkmn,iab->jkamnb',temp_up,temp_down)
#     elif i==int(N/unit_len)-2:
#         for j in range(unit_len):
#             Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             if j==0:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 temp_up = einsum_complex('ijk,ilmn->ljmkn',temp_up,MPOi)
#                 temp_down = complex_conjugate(A)
#             elif j==unit_len-1:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 func_up = einsum_complex('ijk,ilmn->ljmkn',func_up,MPON)
#                 func_down = complex_conjugate(A)
#                 temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],8,sizeA[2],8),einsum_complex('ijkmn,lmnab->iljkab',temp_up,func_up))
#                 temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#             else:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 func_up = einsum_complex('ijk,ilmn->ljmkn',func_up,MPOi)
#                 func_down = complex_conjugate(A)
#                 temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],8,sizeA[2],8),einsum_complex('ijkmn,lmnab->iljkab',temp_up,func_up))
#                 temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#         temp = einsum_complex('ijkmn,iab->jkamnb',temp_up,temp_down)
#         DL = einsum_complex('ijmkln,klnabc->ijmabc',DL,temp)
#     else:
#         for j in range(unit_len):
#             Bk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             if j==0:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 temp_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 temp_up = einsum_complex('ijk,ilmn->ljmkn',temp_up,MPOi)
#                 temp_down = complex_conjugate(A)
#             else:
#                 Bk[0] = torch.tensor(np.cos(-k*(j+unit_len*(i+1)))) * B[0] #- torch.sin(torch.tensor(-k*(i+1))) * B[1]
#                 Bk[1] = torch.tensor(np.sin(-k*(j+unit_len*(i+1)))) * B[0] #+ torch.cos(torch.tensor(-k*(i+1))) * B[1]
#                 func_up = A + einsum_complex('ijk,kn->ijn',Bk,lam_inv)
#                 func_up = einsum_complex('ijk,ilmn->ljmkn',func_up,MPOi)
#                 func_down = complex_conjugate(A)
#                 temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],8,sizeA[2],8),einsum_complex('ijkmn,lmnab->iljkab',temp_up,func_up))
#                 temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#         temp = einsum_complex('ijkmn,iab->jkamnb',temp_up,temp_down)
#         DL = einsum_complex('ijmkln,klnabc->ijmabc',DL,temp)

# print (f"H_size :{size_complex(H_cv)}")
# dev_accu = torch.zeros((sizeA[0]*sizeA[1]*sizeA[2]), dtype=torch.float64)
# for i in range(sizeA[0]*sizeA[1]*sizeA[2]):
#     H_cv[0][i].backward(torch.ones_like(H_cv[0][i]), retain_graph=True)
#     temp = B_grad.grad.view(sizeA[0]*sizeA[1]*sizeA[2]).clone()
#     devr = temp.clone() - dev_accu
#     dev_accu = temp.clone()
#     H_cv[1][i].backward(torch.ones_like(H_cv[1][i]), retain_graph=True)
#     temp = B_grad.grad.view(sizeA[0]*sizeA[1]*sizeA[2]).clone()
#     devi = temp.clone() - dev_accu
#     dev_accu = temp.clone()
#     H_cm[0][i] = devr
#     H_cm[1][i] = devi
# print (size_complex(H_cm))
# #H_cm+= transpose_complex(complex_conjugate(H_cm.clone()))
# #H_cm = H_cm.clone()/2.0
# print (f"H_cm : {H_cm[0][0][1],H_cm[0][1][0],H_cm[1][0][1],H_cm[1][1][0]}")
# #H_cm2 = H_cm.detach().cpu().numpy()
# #H_cmn = H_cm2[0] + 1j*H_cm2[1]
# #e, v = np.linalg.eig(H_cmn)
# #idx = np.argsort(e.real)   
# #e = e[idx]
# #v = v[:,idx]
# #print (e)
# #for i in range(200):
# #    for j in range(200):
# #        if 0.00001 < abs(H_cm[0][i][j]-H_cm[0][j][i]) or 0.00001 < abs(H_cm[1][i][j]+H_cm[1][j][i]) or 0.00001 < abs(H_cm[1][i][i]):
# #            print ('error',i,j,H_cm[0][i][j].item(), H_cm[0][j][i].item(), H_cm[1][i][j].item(), H_cm[1][j][i].item(), H_cm[0][i][i].item(), H_cm[1][i][i].item(),
# #                   abs(H_cm[0][i][j]-H_cm[0][j][i]), abs(H_cm[1][i][j]+H_cm[1][j][i]), abs(H_cm[1][i][i]))
                   

# N_cm2 = N_cm.detach().cpu().numpy()
# N_cmn = N_cm2[0] + 1j*N_cm2[1]
# N_cmn = N_cmn/np.linalg.norm(N_cmn, 'fro')
# N_cmn_inv = linalg.pinvh(N_cmn)#, rcond=0.0)
# #N_cmr = N_cm2[0]
# #N_cmi = N_cm2[1]
# #N_cmr_inv = linalg.inv(N_cmr)
# #N_cmi_inv = linalg.inv(N_cmi)
# #inv_r = linalg.inv(N_cmr + np.matmul(np.matmul(N_cmi,N_cmr_inv),N_cmi))
# #inv_i = -linalg.inv(N_cmi + np.matmul(np.matmul(N_cmr,N_cmi_inv),N_cmr))
# #N_cmn_inv = inv_r + 1j*inv_i
# H_cm2 = H_cm.detach().cpu().numpy()
# H_cmn = H_cm2[0] + 1j*H_cm2[1]
# H_cmn = H_cmn/np.linalg.norm(H_cmn, 'fro')
# print (H_cmn[0][1].real, H_cmn[1][0].real, H_cmn[0][1].imag, H_cmn[1][0].imag)
# #for i in range(200):
# #    for j in range(200):
# #        if 0.00001 < abs(H_cmn[i][j].real-H_cmn[j][i].real) or 0.00001 < abs(H_cmn[i][j].imag+H_cmn[j][i].imag) or 0.00001 < abs(H_cmn[i][i].imag):
# #            print ('error',H_cmn[i][j].real, H_cmn[j][i].real, H_cmn[i][j].imag, H_cmn[j][i].imag, H_cmn[i][i].real, H_cmn[i][i].imag)
# print (N_cmn[0][1].real, N_cmn[1][0].real, N_cmn[0][1].imag, N_cmn[1][0].imag)
# #for i in range(200):
# #    for j in range(200):
# #        if 0.00001 < abs(N_cmn[i][j].real-N_cmn[j][i].real) or 0.00001 < abs(N_cmn[i][j].imag+N_cmn[j][i].imag) or 0.00001 < abs(N_cmn[i][i].imag):
# #            print ('error',N_cmn[i][j].real, N_cmn[j][i].real, N_cmn[i][j].imag, N_cmn[j][i].imag, N_cmn[i][i].real, N_cmn[i][i].imag)
# print (N_cmn_inv[0][1].real, N_cmn_inv[1][0].real, N_cmn_inv[0][1].imag, N_cmn_inv[1][0].imag)
# #for i in range(200):
# #    for j in range(200):
# #        if 0.00001 < abs(N_cmn_inv[i][j].real-N_cmn_inv[j][i].real) or 0.00001 < abs(N_cmn_inv[i][j].imag+N_cmn_inv[j][i].imag) or 0.00001 < abs(N_cmn_inv[i][i].imag):
# #            print ('error',N_cmn_inv[i][j].real, N_cmn_inv[j][i].real, N_cmn_inv[i][j].imag, N_cmn_inv[j][i].imag, N_cmn_inv[i][i].real, N_cmn_inv[i][i].imag)
# #for i in range(200):
# #    for j in range(200):
# #        if 0.001 < abs(np.matmul(N_cmn_inv,H_cmn)[i][j].real-np.matmul(N_cmn_inv,H_cmn)[j][i].real) and 0.001 < abs(np.matmul(N_cmn_inv,H_cmn)[i][j].imag+np.matmul(N_cmn_inv,H_cmn)[j][i].imag):
# #            print ('error',np.matmul(N_cmn_inv,H_cmn)[i][j].real, np.matmul(N_cmn_inv,H_cmn)[j][i].real, np.matmul(N_cmn_inv,H_cmn)[i][j].imag, np.matmul(N_cmn_inv,H_cmn)[j][i].imag)
# e, v = np.linalg.eig(np.matmul(N_cmn_inv,H_cmn))#, numofstate, sigma=-1.0, which='SM')
# idx = np.argsort(e)   
# e = e[idx]
# v = v[:,idx]
# #for i in range(e.shape[0]):
# #    if (abs(e[i].imag)<0.001):
# #        print (e[i].real)
# print (e)

# ################MPO################
# with torch.no_grad():
#     iden = torch.zeros((2,model.phys_dim,model.phys_dim), dtype=torch.float64)
#     iden[0] = torch.eye(model.phys_dim, dtype=torch.float64)
#     iden2 = einsum_complex('ij,kl->ikjl',iden,iden)
#     iden3 = einsum_complex('ikjl,ab->ikajlb',iden2,iden)
#     ssx = model.SSx
#     ssz = model.SSz
#     sx = model.Sx
#     sz = model.Sz
#     xi = einsum_complex('ij,kl->ikjl',sx,iden)
#     xix = einsum_complex('ikjl,ab->ikajlb',xi,sx)
#     zi = einsum_complex('ij,kl->ikjl',sz,iden)
#     ziz = einsum_complex('ikjl,ab->ikajlb',zi,sz)
# ################Excited state################
# # e = np.zeros((2*sizeA[0]*sizeA[1]*sizeA[2]), dtype=np.complex128)
# # v = np.zeros((2*sizeA[0]*sizeA[1]*sizeA[2],2*sizeA[0]*sizeA[1]*sizeA[2]), dtype=np.complex128)
# e = np.zeros((2*sizeC[0]*sizeC[1]*sizeC[2]), dtype=np.complex128)
# v = np.zeros((2*sizeC[0]*sizeC[1]*sizeC[2],2*sizeC[0]*sizeC[1]*sizeC[2]), dtype=np.complex128)

# # filename1 = 'state_'+str(numk)+'_9.txt'
# # with open(filename1, 'r') as file_to_read2:
# #     lines2 = file_to_read2.readline()
# #     q_tmp = [complex(k) for k in lines2.replace('i', 'j').split()]
# #     for i in range(2*sizeA[0]*sizeA[1]*sizeA[2]):
# #         for j in range(2*sizeA[0]*sizeA[1]*sizeA[2]):
# #             v[i][j]=q_tmp[i*2*sizeA[0]*sizeA[1]*sizeA[2]+j]

# # filename1 = 'energy_'+str(numk)+'_9.txt'
# # with open(filename1, 'r') as file_to_read2:
# #     lines2 = file_to_read2.readline()
# #     q_tmp = [complex(k) for k in lines2.replace('i', 'j').split()]
# #     for i in range(2*sizeA[0]*sizeA[1]*sizeA[2]):
# #         e[i]=q_tmp[i]

# Vec1 = []
# Vec2 = []
# for i in range(numofstate):
#     Vec1.append(np.tensordot(v[:sizeA[0]*sizeA[1]*sizeA[2],i].reshape(sizeA[0], sizeA[1], sizeA[2]),np.diag(1/C0),([2],[0])))
#     Vec2.append(np.tensordot(v[sizeA[0]*sizeA[1]*sizeA[2]:2*sizeA[0]*sizeA[1]*sizeA[2],i].reshape(sizeA[0], sizeA[1], sizeA[2]),np.diag(1/C1),([2],[0])))

# ################Energy################
# norm = []
# ener = []
# torch.autograd.set_detect_anomaly(True)
# for i in range(numofstate):
#     lamb = torch.tensor(0.0).requires_grad_(True)
#     Vr = torch.as_tensor(Vec1[i].real)
#     Vi = torch.as_tensor(Vec1[i].imag)
#     V1 = torch.stack((Vr, Vi), dim=0)
#     Vr = torch.as_tensor(Vec2[i].real)
#     Vi = torch.as_tensor(Vec2[i].imag)
#     V2 = torch.stack((Vr, Vi), dim=0)
#     nor = torch.tensor(0.0)
#     for n in range(int(N/unit_len)):
#         if n==0:
#             for j in range(unit_len):
#                 Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#                 if j==0:
#                     Vk[0] = torch.tensor(np.cos(-2*k*(int(j/2)))) * V1[0] - torch.tensor(np.sin(-2*k*(int(j/2)))) * V1[1]
#                     Vk[1] = torch.tensor(np.sin(-2*k*(int(j/2)))) * V1[0] + torch.tensor(np.cos(-2*k*(int(j/2)))) * V1[1]
#                     temp_up = A1 + lamb * Vk
#                     temp_down = complex_conjugate(V1)
#                     temp_down2 = complex_conjugate(A1)
#                 elif j==1:
#                     Vk[0] = torch.tensor(np.cos(-2*k*(int(j/2)))) * V2[0] - torch.tensor(np.sin(-2*k*(int(j/2)))) * V2[1]
#                     Vk[1] = torch.tensor(np.sin(-2*k*(int(j/2)))) * V2[0] + torch.tensor(np.cos(-2*k*(int(j/2)))) * V2[1]
#                     func_up = A2 + lamb * Vk
#                     func_down = complex_conjugate(A2)
#                     func_down2 = complex_conjugate(V2)
#                     temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                     temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#                     temp_down2 = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down2,func_down2))
#                 else:
#                     if j%2 == 0:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int(j/2)))) * V1[0] - torch.tensor(np.sin(-2*k*(int(j/2)))) * V1[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int(j/2)))) * V1[0] + torch.tensor(np.cos(-2*k*(int(j/2)))) * V1[1]
#                         func_up = A1 + lamb * Vk
#                         func_down = complex_conjugate(A1)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#                         temp_down2 = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down2,func_down))
#                     else:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int(j/2)))) * V2[0] - torch.tensor(np.sin(-2*k*(int(j/2)))) * V2[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int(j/2)))) * V2[0] + torch.tensor(np.cos(-2*k*(int(j/2)))) * V2[1]
#                         func_up = A2 + lamb * Vk
#                         func_down = complex_conjugate(A2)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#                         temp_down2 = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down2,func_down))
#             DL = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
#             DL2 = einsum_complex('ijk,imn->jmkn',temp_up,temp_down2)
#         elif n==int(N/unit_len)-1:
#             for j in range(unit_len):
#                 Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#                 if j==0:
#                     Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V1[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V1[1]
#                     Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V1[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V1[1]
#                     temp_up = A1 + lamb * Vk
#                     temp_down = complex_conjugate(A1)
#                 elif j==1:
#                     Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V2[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V2[1]
#                     Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V2[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V2[1]
#                     func_up = A2 + lamb * Vk
#                     func_down = complex_conjugate(A2)
#                     temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                     temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#                 else:
#                     if j%2 == 0:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V1[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V1[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V1[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V1[1]
#                         func_up = A1 + lamb * Vk
#                         func_down = complex_conjugate(A1)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#                     else:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V2[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V2[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V2[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V2[1]
#                         func_up = A2 + lamb * Vk
#                         func_down = complex_conjugate(A2)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#             temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
#             nor = einsum_complex('ijkl,klij',DL,temp)[0]
#             nor+= einsum_complex('ijkl,klij',DL2,temp)[0]
#         else:
#             for j in range(unit_len):
#                 Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#                 if j==0:
#                     Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V1[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V1[1]
#                     Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V1[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V1[1]
#                     temp_up = A1 + lamb * Vk
#                     temp_down = complex_conjugate(A1)
#                 elif j==1:
#                     Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V2[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V2[1]
#                     Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V2[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V2[1]
#                     func_up = A2 + lamb * Vk
#                     func_down = complex_conjugate(A2)
#                     temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                     temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#                 else:
#                     if j%2 == 0:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V1[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V1[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V1[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V1[1]
#                         func_up = A1 + lamb * Vk
#                         func_down = complex_conjugate(A1)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#                     else:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V2[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V2[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2)))) * V2[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2)))) * V2[1]
#                         func_up = A2 + lamb * Vk
#                         func_down = complex_conjugate(A2)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#             temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
#             DL = einsum_complex('ijkl,klmn->ijmn',DL,temp)
#             DL2 = einsum_complex('ijkl,klmn->ijmn',DL2,temp)
#     nor.backward()
#     norm.append(lamb.grad.item())

#     lamb = torch.tensor(0.0).requires_grad_(True)
#     lame = torch.tensor(0.0).requires_grad_(True)
#     h = iden2 + lame * (-Jx * ssx - Jz * ssz)
#     h_nnn = iden3 + lame * (Kx * xix + Kz * ziz)
#     ene = torch.tensor(0.0)
#     up = []
#     for n in range(int(N/2)):
#         Vk1 = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#         Vk2 = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#         Vk1[0] = torch.tensor(np.cos(-2*k*(n))) * V1[0] - torch.tensor(np.sin(-2*k*(n))) * V1[1]
#         Vk1[1] = torch.tensor(np.sin(-2*k*(n))) * V1[0] + torch.tensor(np.cos(-2*k*(n))) * V1[1]
#         Vk2[0] = torch.tensor(np.cos(-2*k*(n))) * V2[0] - torch.tensor(np.sin(-2*k*(n))) * V2[1]
#         Vk2[1] = torch.tensor(np.sin(-2*k*(n))) * V2[0] + torch.tensor(np.cos(-2*k*(n))) * V2[1]
#         mps1 = A1 + lamb * Vk1
#         mps2 = A2 + lamb * Vk2
#         theta = einsum_complex('ijk,lkm->ijlm',mps1,mps2)
#         theta_O = einsum_complex('ijkl,ikab->abjl',theta,h)
#         up.append(theta_O)
#     theta = einsum_complex('ijk,lkm->ijlm',complex_conjugate(V2),complex_conjugate(A1))
#     theta_O1 = einsum_complex('ijkl,kalb->ijab',h,theta)
#     theta = einsum_complex('ijk,lkm->ijlm',complex_conjugate(A2),complex_conjugate(A1))
#     theta_Oi = einsum_complex('ijkl,kalb->ijab',h,theta)
#     for i in range(int(N/2)-1):
#         if i==0:
#             temp = einsum_complex('ijkl,jmno->imknlo',up[i],theta_O1)
#             DL = einsum_complex('ijklmn,jamb->iaklbn',temp,up[i+1])
#         elif i==int(N/2)-1-1:
#             temp = einsum_complex('ijklmn,janb->iaklmb',DL,theta_Oi)
#             DL = einsum_complex('ijklmn,jamk->ialn',temp,up[i+1])
#         else:
#             temp = einsum_complex('ijklmn,janb->iaklmb',DL,theta_Oi)
#             DL = einsum_complex('ijklmn,jamb->iaklbn',temp,up[i+1])
#     DL = einsum_complex('ijkl,jiab->abkl',DL,h)
#     DL = contiguous_complex(einsum_complex('ijkl,ilm->jmk',DL,complex_conjugate(A2)))
#     ene = contiguous_complex(einsum_complex('ijk,ijk',DL,complex_conjugate(A1)))[0]

#     theta = einsum_complex('ijk,lkm->ijlm',complex_conjugate(A2),complex_conjugate(A1))
#     theta_O1 = einsum_complex('ijkl,kalb->ijab',h,theta)
#     theta = einsum_complex('ijk,lkm->ijlm',complex_conjugate(A2),complex_conjugate(A1))
#     theta_Oi = einsum_complex('ijkl,kalb->ijab',h,theta)
#     for i in range(int(N/2)-1):
#         if i==0:
#             temp = einsum_complex('ijkl,jmno->imknlo',up[i],theta_O1)
#             DL = einsum_complex('ijklmn,jamb->iaklbn',temp,up[i+1])
#         elif i==int(N/2)-1-1:
#             temp = einsum_complex('ijklmn,janb->iaklmb',DL,theta_Oi)
#             DL = einsum_complex('ijklmn,jamk->ialn',temp,up[i+1])
#         else:
#             temp = einsum_complex('ijklmn,janb->iaklmb',DL,theta_Oi)
#             DL = einsum_complex('ijklmn,jamb->iaklbn',temp,up[i+1])
#     DL = einsum_complex('ijkl,jiab->abkl',DL,h)
#     DL = contiguous_complex(einsum_complex('ijkl,ilm->jmk',DL,complex_conjugate(A2)))
#     ene+= contiguous_complex(einsum_complex('ijk,ijk',DL,complex_conjugate(V1)))[0]

#     up = []
#     for n in range(int(N/3)):
#         if n%2 == 0:
#             Vk1 = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             Vk2 = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             Vk3 = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             Vk1[0] = torch.tensor(np.cos(-2*k*int((n*3)/2))) * V1[0] - torch.tensor(np.sin(-2*k*int((n*3)/2))) * V1[1]
#             Vk1[1] = torch.tensor(np.sin(-2*k*int((n*3)/2))) * V1[0] + torch.tensor(np.cos(-2*k*int((n*3)/2))) * V1[1]
#             Vk2[0] = torch.tensor(np.cos(-2*k*int((n*3+1)/2))) * V2[0] - torch.tensor(np.sin(-2*k*int((n*3+1)/2))) * V2[1]
#             Vk2[1] = torch.tensor(np.sin(-2*k*int((n*3+1)/2))) * V2[0] + torch.tensor(np.cos(-2*k*int((n*3+1)/2))) * V2[1]
#             Vk3[0] = torch.tensor(np.cos(-2*k*int((n*3+2)/2))) * V1[0] - torch.tensor(np.sin(-2*k*int((n*3+2)/2))) * V1[1]
#             Vk3[1] = torch.tensor(np.sin(-2*k*int((n*3+2)/2))) * V1[0] + torch.tensor(np.cos(-2*k*int((n*3+2)/2))) * V1[1]
#             mps1 = A1 + lamb * Vk1
#             mps2 = A2 + lamb * Vk2
#             mps3 = A1 + lamb * Vk3
#             theta = einsum_complex('ijk,lkm->ijlm',mps1,mps2)
#             theta = einsum_complex('ijlm,kmn->ilkjn',theta,mps3)
#             theta_O = einsum_complex('ijklm,ijkabc->abclm',theta,h_nnn)
#             theta_O = einsum_complex('ijklm,jknabc->iabcnlm',theta_O,h_nnn)
#             up.append(theta_O)
#         else:
#             Vk1 = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             Vk2 = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             Vk3 = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#             Vk1[0] = torch.tensor(np.cos(-2*k*int((n*3)/2))) * V2[0] - torch.tensor(np.sin(-2*k*int((n*3)/2))) * V2[1]
#             Vk1[1] = torch.tensor(np.sin(-2*k*int((n*3)/2))) * V2[0] + torch.tensor(np.cos(-2*k*int((n*3)/2))) * V2[1]
#             Vk2[0] = torch.tensor(np.cos(-2*k*int((n*3+1)/2))) * V1[0] - torch.tensor(np.sin(-2*k*int((n*3+1)/2))) * V1[1]
#             Vk2[1] = torch.tensor(np.sin(-2*k*int((n*3+1)/2))) * V1[0] + torch.tensor(np.cos(-2*k*int((n*3+1)/2))) * V1[1]
#             Vk3[0] = torch.tensor(np.cos(-2*k*int((n*3+2)/2))) * V2[0] - torch.tensor(np.sin(-2*k*int((n*3+2)/2))) * V2[1]
#             Vk3[1] = torch.tensor(np.sin(-2*k*int((n*3+2)/2))) * V2[0] + torch.tensor(np.cos(-2*k*int((n*3+2)/2))) * V2[1]
#             mps1 = A2 + lamb * Vk1
#             mps2 = A1 + lamb * Vk2
#             mps3 = A2 + lamb * Vk3
#             theta = einsum_complex('ijk,lkm->ijlm',mps1,mps2)
#             theta = einsum_complex('ijlm,kmn->ilkjn',theta,mps3)
#             theta_O = einsum_complex('ijklm,ijkabc->abclm',theta,h_nnn)
#             theta_O = einsum_complex('ijklm,jknabc->iabcnlm',theta_O,h_nnn)
#             up.append(theta_O)
#     theta = einsum_complex('ijk,lkm->ijlm',complex_conjugate(A2),complex_conjugate(A1))
#     theta = einsum_complex('ijlm,kmn->ilkjn',theta,complex_conjugate(A2))
#     theta_Oi2 = einsum_complex('ijklmn,lmnab->ijkab',h_nnn,theta)
#     theta = einsum_complex('ijk,lkm->ijlm',complex_conjugate(A1),complex_conjugate(A2))
#     theta = einsum_complex('ijlm,kmn->ilkjn',theta,complex_conjugate(A1))
#     theta_Oi = einsum_complex('ijklmn,lmnab->ijkab',h_nnn,theta)
#     for i in range(int(N/3)-1):
#         if i==0:
#             temp = einsum_complex('ijklabc,klmno->ijambnco',up[i],theta_Oi)
#             DL = einsum_complex('ijklmnab,klopqac->ijopqmncb',temp,up[i+1])
#         elif i==int(N/3)-1-1:
#             temp = einsum_complex('ijklmnabc,klpcd->ijmpnabd',DL,theta_Oi)
#             DL = einsum_complex('ijklmnab,klopiam->jopnb',temp,up[i+1])
#         else:
#             if i%2 ==0:
#                 temp = einsum_complex('ijklmnabc,klpcd->ijmpnabd',DL,theta_Oi)
#                 DL = einsum_complex('ijklmnab,klopqac->ijopqmncb',temp,up[i+1])
#             else:
#                 temp = einsum_complex('ijklmnabc,klpcd->ijmpnabd',DL,theta_Oi2)
#                 DL = einsum_complex('ijklmnab,klopqac->ijopqmncb',temp,up[i+1])
#     DL1 = einsum_complex('ijklm,jkiabc->abclm',DL,h_nnn)
#     DL1 = contiguous_complex(einsum_complex('ijklm,imn->jkln',DL1,complex_conjugate(A2)))
#     DL1 = contiguous_complex(einsum_complex('ijkl,jmk->ilm',DL1,complex_conjugate(A2)))
#     ene+= contiguous_complex(einsum_complex('ijk,ijk',DL1,complex_conjugate(V1)))[0]

#     DL2 = einsum_complex('ijklm,jkiabc->abclm',DL,h_nnn)
#     DL2 = contiguous_complex(einsum_complex('ijklm,imn->jkln',DL2,complex_conjugate(A2)))
#     DL2 = contiguous_complex(einsum_complex('ijkl,jmk->ilm',DL2,complex_conjugate(V2)))
#     ene+= contiguous_complex(einsum_complex('ijk,ijk',DL2,complex_conjugate(A1)))[0]

#     g = torch.autograd.grad(ene,lamb,create_graph=True)      
#     g[0].backward()
#     ener.append(lame.grad.item())

# for i in range(numofstate):
#     print (ener[i]/norm[i]/N)

# ################Dynamical structural factor################
# sz = torch.zeros((2,2,2), dtype=torch.float64)
# sz[0][0][0]=0.5
# sz[0][1][1]=-0.5
# iden = torch.zeros((2,2,2), dtype=torch.float64)
# iden[0][0][0]=1.0
# iden[0][1][1]=1.0

# mat_ele=[]
# norm_k=[]

# nor_0 = view_complex((sizeA[1]**2, sizeA[2]**2),einsum_complex('ijk,imn->jmkn',A1,complex_conjugate(A1)))
# functional_DL = view_complex((sizeA[1]**2, sizeA[2]**2),einsum_complex('ijk,imn->jmkn',A2,complex_conjugate(A2)))
# nor_0 = mm_complex(nor_0,functional_DL)
# for i in range(int(N/2)-1):
#     functional_DL = view_complex((sizeA[1]**2, sizeA[2]**2),einsum_complex('ijk,imn->jmkn',A1,complex_conjugate(A1)))
#     nor_0 = mm_complex(nor_0,functional_DL)
#     functional_DL = view_complex((sizeA[1]**2, sizeA[2]**2),einsum_complex('ijk,imn->jmkn',A2,complex_conjugate(A2)))
#     nor_0 = mm_complex(nor_0,functional_DL)
# norm_0 = torch.trace(nor_0[0]).item()

# for i in range(numofstate):
#     Vr = torch.as_tensor(Vec1[i].real, dtype=torch.float64)
#     Vi = torch.as_tensor(Vec1[i].imag, dtype=torch.float64)
#     V1 = torch.stack((Vr, Vi), dim=0)
#     Vr = torch.as_tensor(Vec2[i].real, dtype=torch.float64)
#     Vi = torch.as_tensor(Vec2[i].imag, dtype=torch.float64)
#     V2 = torch.stack((Vr, Vi), dim=0)

#     lamb = torch.tensor(0.0, dtype=torch.float64).requires_grad_(True)
#     lams = torch.tensor(0.0, dtype=torch.float64).requires_grad_(True)
#     specr = torch.tensor(0.0, dtype=torch.float64)
#     speci = torch.tensor(0.0, dtype=torch.float64)
#     idenr = torch.eye(sizeA[0], dtype=torch.float64)
#     ideni = torch.zeros((sizeA[0],sizeA[0]), dtype=torch.float64)
#     iden = torch.stack((idenr, ideni), dim=0)
#     for n in range(int(N/unit_len)):
#         if n==0:
#             for j in range(unit_len):
#                 Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#                 sk = torch.zeros((2,2,2), dtype=torch.float64)
#                 if j==0:
#                     Vk[0] = torch.tensor(np.cos(-2*k*(int(j/2))), dtype=torch.float64) * V1[0] - torch.tensor(np.sin(-2*k*(int(j/2))), dtype=torch.float64) * V1[1]
#                     Vk[1] = torch.tensor(np.sin(-2*k*(int(j/2))), dtype=torch.float64) * V1[0] + torch.tensor(np.cos(-2*k*(int(j/2))), dtype=torch.float64) * V1[1]
#                     temp_up = A1 + lamb * Vk
#                     sk[0] = torch.tensor(np.cos(2*k*(int(j/2))), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(2*k*(int(j/2))), dtype=torch.float64) * sz[1]
#                     sk[1] = torch.tensor(np.sin(2*k*(int(j/2))), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(2*k*(int(j/2))), dtype=torch.float64) * sz[1]
#                     temp_up = einsum_complex('ijk,il->ljk',temp_up,(iden + lams * sk))
#                     temp_down = complex_conjugate(A1)
#                 else:
#                     if j%2 == 0:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int(j/2))), dtype=torch.float64) * V1[0] - torch.tensor(np.sin(-2*k*(int(j/2))), dtype=torch.float64) * V1[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int(j/2))), dtype=torch.float64) * V1[0] + torch.tensor(np.cos(-2*k*(int(j/2))), dtype=torch.float64) * V1[1]
#                         func_up = A1 + lamb * Vk
#                         sk[0] = torch.tensor(np.cos(2*k*(int(j/2))), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(2*k*(int(j/2))), dtype=torch.float64) * sz[1]
#                         sk[1] = torch.tensor(np.sin(2*k*(int(j/2))), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(2*k*(int(j/2))), dtype=torch.float64) * sz[1]
#                         func_up = einsum_complex('ijk,il->ljk',func_up,(iden + lams * sk))
#                         func_down = complex_conjugate(A1)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#                     else:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int(j/2))), dtype=torch.float64) * V2[0] - torch.tensor(np.sin(-2*k*(int(j/2))), dtype=torch.float64) * V2[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int(j/2))), dtype=torch.float64) * V2[0] + torch.tensor(np.cos(-2*k*(int(j/2))), dtype=torch.float64) * V2[1]
#                         func_up = A2 + lamb * Vk
#                         sk[0] = torch.tensor(np.cos(2*k*(int(j/2))), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(2*k*(int(j/2))), dtype=torch.float64) * sz[1]
#                         sk[1] = torch.tensor(np.sin(2*k*(int(j/2))), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(2*k*(int(j/2))), dtype=torch.float64) * sz[1]
#                         func_up = einsum_complex('ijk,il->ljk',func_up,(iden + lams * sk))
#                         func_down = complex_conjugate(A2)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#             DL = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
#         elif n==int(N/unit_len)-1:
#             for j in range(unit_len):
#                 Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#                 sk = torch.zeros((2,2,2), dtype=torch.float64)
#                 if j==0:
#                     Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[1]
#                     Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[1]
#                     temp_up = A1 + lamb * Vk
#                     sk[0] = torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                     sk[1] = torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                     temp_up = einsum_complex('ijk,il->ljk',temp_up,(iden + lams * sk))
#                     temp_down = complex_conjugate(A1)
#                 else:
#                     if j%2 == 0:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[1]
#                         func_up = A1 + lamb * Vk
#                         sk[0] = torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                         sk[1] = torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                         func_up = einsum_complex('ijk,il->ljk',func_up,(iden + lams * sk))
#                         func_down = complex_conjugate(A1)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#                     else:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V2[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V2[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V2[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V2[1]
#                         func_up = A2 + lamb * Vk
#                         sk[0] = torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                         sk[1] = torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                         func_up = einsum_complex('ijk,il->ljk',func_up,(iden + lams * sk))
#                         func_down = complex_conjugate(A2)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))        
#             temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
#             DL = einsum_complex('ijkl,klij',DL,temp)
#             specr = DL[0]
#             speci = DL[1]
#         else:
#             for j in range(unit_len):
#                 Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
#                 sk = torch.zeros((2,2,2), dtype=torch.float64)
#                 if j==0:
#                     Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[1]
#                     Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[1]
#                     temp_up = A1 + lamb * Vk
#                     sk[0] = torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                     sk[1] = torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                     temp_up = einsum_complex('ijk,il->ljk',temp_up,(iden + lams * sk))
#                     temp_down = complex_conjugate(A1)
#                 else:
#                     if j%2 == 0:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V1[1]
#                         func_up = A1 + lamb * Vk
#                         sk[0] = torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                         sk[1] = torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                         func_up = einsum_complex('ijk,il->ljk',func_up,(iden + lams * sk))
#                         func_down = complex_conjugate(A1)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#                     else:
#                         Vk[0] = torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V2[0] - torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V2[1]
#                         Vk[1] = torch.tensor(np.sin(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V2[0] + torch.tensor(np.cos(-2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * V2[1]
#                         func_up = A2 + lamb * Vk
#                         sk[0] = torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                         sk[1] = torch.tensor(np.sin(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(2*k*(int((j+unit_len*n)/2))), dtype=torch.float64) * sz[1]
#                         func_up = einsum_complex('ijk,il->ljk',func_up,(iden + lams * sk))
#                         func_down = complex_conjugate(A2)
#                         temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
#                         temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
#             temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
#             DL = einsum_complex('ijkl,klmn->ijmn',DL,temp)

#     g = torch.autograd.grad(specr,lamb,create_graph=True)
#     g[0].backward(retain_graph=True)
#     devr = lams.grad.clone()

#     g = torch.autograd.grad(speci,lamb,create_graph=True)
#     g[0].backward(retain_graph=True)
#     devi = lams.grad.clone() - devr
    
#     matr = devr.detach().cpu().numpy()
#     mati = devi.detach().cpu().numpy()
#     mat_ele.append(np.abs(matr+1j*mati))



# ene_final = []
# mat_ele_final = []
# for i in range(numofstate):
#     ene_final.append(ener[i]/norm[i]/N)
#     mat_ele_final.append(mat_ele[i]**2/abs(norm[i])/abs(norm_0)/N/N)
#     print (ene_final[-1], mat_ele_final[-1])

# str1 = " ".join(str(e) for e in ene_final)
# f=open('ene_'+str(numk)+'_9.txt', 'w')
# f.write(str1)
# f.close()

# str1 = " ".join(str(e) for e in mat_ele_final)
# f=open('mat_ele_'+str(numk)+'_9.txt', 'w')
# f.write(str1)
# f.close()

# tEnd = time.time()
# print (tEnd - tStart)
