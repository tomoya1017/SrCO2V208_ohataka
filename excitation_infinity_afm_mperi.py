import context
import time
import torch
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse
import config as cfg
import pickle
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
N = 10
1.0000000000000042
1.0000000000000524
1.0000000000004898
1.0000000000004898

0.9999999999999818
0.9999999999998312
0.9999999999984108
0.9999999999841007
ex_bx = 0
ex_by = ex_bx*0.29
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

def load_from_pickle(filename):
    """
    pickle ファイルから Tn と lam を復元
    - filename: 保存された pickle ファイルの名前
    - 戻り値: Tn (リスト形式で [Tn[0], Tn[1], ...], 各要素が (2, 3, 10, 10)) と lam
    """
    # pickle ファイルを読み込み
    with open(filename, "rb") as file:
        data = pickle.load(file)

    # Tn はそのままの形式で返す
    return data["Tn"], data["lam"]


filename = "/home/23_takahashi/b{}_chi10_imag.pkl".format(ex_bx)
loaded_Tn, loaded_lam = load_from_pickle(filename)

################Left Canonical################
A2 = loaded_Tn[0]
A = A2[0] + 1j*A2[1]
sizeA = A.shape
print(sizeA)

lamA2 = loaded_lam[0]
lamA = lamA2[0] + 1j*lamA2[1]
size_lamA = lamA.shape
print(size_lamA)

B2 = loaded_Tn[1]
B = B2[0] + 1j*B2[1]
sizeB = B.shape
print(sizeB)

lamB2 = loaded_lam[1]
lamB = lamB2[0] + 1j*lamB2[1]
size_lamB = lamB.shape
print(size_lamB)

########check canonical environment##########
# A_check = np.tensordot(np.tensordot(np.tensordot(np.diag(lamA),np.diag(lamA),(0,0)),A,(0,1)),A.conj(),([0,1],[1,0]))
# print(f"A_check;{A_check}")
# A_check2 = np.tensordot(np.tensordot(np.tensordot(A,np.diag(lamB),(2,0)),np.diag(lamB),(2,0)),A.conj(),([0,2],[0,2]))
# print(f"A_check2;{A_check2}")

# B_check = np.tensordot(np.tensordot(np.tensordot(np.diag(lamB),np.diag(lamB),(0,0)),B,(0,1)),B.conj(),([0,1],[1,0]))
# print(f"B_check;{B_check}")

# B_check2 = np.tensordot(np.tensordot(np.tensordot(B,np.diag(lamA),(2,0)),np.diag(lamA),(2,0)),B.conj(),([0,2],[0,2]))
# print(f"B_check2;{B_check2}")

C1 = np.transpose(np.tensordot(np.tensordot(np.tensordot(np.diag(lamB),B,(1,1)),np.diag(lamA),(2,0)),A,(2,1)),(2,0,1,3)).reshape(sizeA[0]**2,sizeA[1],sizeA[2])
# C1 = np.transpose(np.tensordot(A,B,(2,1)),(0,2,1,3)).reshape(sizeA[0]**2,sizeA[1],sizeA[2])
print(f"len:{len(loaded_lam)}")


# C_check = np.tensordot(np.tensordot(np.tensordot(np.diag(lamA),np.diag(lamA),(0,0)),C1,(0,1)),C1.conj(),([0,1],[1,0]))
# print(f"C_check;{C_check}")
# C_check2 = np.tensordot(np.tensordot(np.tensordot(C1,np.diag(lamA),(2,0)),np.diag(lamA),(2,0)),C1.conj(),([0,2],[0,2]))
# print(f"C_check2;{C_check2}")

C2 = np.transpose(np.tensordot(np.tensordot(np.tensordot(A,np.diag(lamB),(2,0)),B,(2,1)),np.diag(lamA),(3,0)),(0,2,1,3)).reshape(sizeA[0]**2,sizeA[1],sizeA[2])
# # C1 = np.transpose(np.tensordot(A,B,(2,1)),(0,2,1,3)).reshape(sizeA[0]**2,sizeA[1],sizeA[2])
# print(f"len:{len(loaded_lam)}")


# # C_check = np.tensordot(np.tensordot(np.tensordot(np.diag(lamA),np.diag(lamA),(0,0)),C1,(0,1)),C1.conj(),([0,1],[1,0]))
# # print(f"C_check;{C_check}")
# C_check2 = np.tensordot(np.tensordot(np.tensordot(C1,np.diag(lamB),(2,0)),np.diag(lamB),(2,0)),C1.conj(),([0,2],[0,2]))
# print(f"C_check2;{C_check2}")


## assume MPS is in canonical form


# El = np.identity(size_lamB[0])
El = np.tensordot(np.tensordot(np.identity(size_lamB[0]),np.diag(lamB),(1,0)),np.diag(lamB),(0,0))
# print(size_lamB[0])
# # El = np.tensordot(np.tensordot(El,C1,(1,1)),C1.conj(),([0,1],[1,0]))
El = El.reshape(size_lamB[0]**2)
# # print(El)
Er = np.identity(size_lamB[0])
Er = Er.reshape((size_lamB[0]**2))
# print(Er.shape)
# Env_left.append(np.dot(np.dot(np.diag(loaded_lam[0]),np.identity((loaded_lam[0].shape[0]))),np.diag(loaded_lam[0])))
# Env_right.append(np.dot(np.dot(np.diag(loaded_lam[1]),np.identity((loaded_lam[1].shape[0]))),np.diag(loaded_lam[1])))
# print(f"left_shape:{Env_left[0].shape}")
# print(f"right_shape:{Env_right[0].shape}")

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


AL, L, lam = leftorthonormalize(C2, L0)
AR, C, lam = rightorthonormalize(AL, L0)
u, C, v = linalg.svd(C)
AL = np.tensordot(np.conjugate(np.transpose(u,(1,0))), AL, (1,1))
AL = np.transpose(np.tensordot(AL, u, (2,0)),(1,0,2))
AR = np.tensordot(np.conjugate(np.transpose(v,(1,0))), AR, (1,1))
AR = np.transpose(np.tensordot(AR, v, (2,0)),(1,0,2))
AC = np.tensordot(AL,np.diag(C),(2,0))
print(f"AC:{C.shape}")
ALr = torch.as_tensor(AL.real, dtype=torch.float64)
ALi = torch.as_tensor(AL.imag, dtype=torch.float64)
AL = torch.stack((ALr, ALi), dim=0)

ARr = torch.as_tensor(AR.real, dtype=torch.float64)
ARi = torch.as_tensor(AR.imag, dtype=torch.float64)
AR = torch.stack((ARr, ARi), dim=0)

ACr = torch.as_tensor(AC.real, dtype=torch.float64)
ACi = torch.as_tensor(AC.imag, dtype=torch.float64)
AC = torch.stack((ACr, ACi), dim=0)
print(f"AC:{AC.shape}")

Cr = torch.as_tensor(C2.real, dtype=torch.float64)
Ci = torch.as_tensor(C2.imag, dtype=torch.float64)
C2 = torch.stack((Cr, Ci), dim=0)

# El = contiguous_complex(einsum_complex('ijk,ijl->kl', AL, complex_conjugate(AL)))
# El = view_complex((sizeA[1]**2),El)
# Er = contiguous_complex(einsum_complex('ijk,ilk->jl', AR, complex_conjugate(AR)))
# Er = view_complex((sizeA[2]**2),Er)


Elr = torch.as_tensor(El.real.copy(), dtype=torch.float64)
Eli = torch.as_tensor(El.imag.copy(), dtype=torch.float64)
El = torch.stack((Elr, Eli), dim=0)
print(f"el:{El.shape}")
Err = torch.as_tensor(Er.real.copy(), dtype=torch.float64)
Eri = torch.as_tensor(Er.imag.copy(), dtype=torch.float64)
Er = torch.stack((Err, Eri), dim=0)
print(f"er:{Er.shape}")


# A_DL = np.transpose(np.tensordot(AL, np.conjugate(AL), (0,0)), (1,3,0,2)).reshape(sizeA[1]**2, sizeA[2]**2)
# #print (spr_linalg.eigs(A_DL, 1, which='LM'))
# #print (C)
# lamr = torch.as_tensor(np.diag(1/C).real, dtype=torch.float64)
# lami = torch.zeros((lamr.size()[0],lamr.size()[1]), dtype=torch.float64)
# lam_inv = torch.stack((lamr, lami), dim=0)
#print (sizeA[0]*sizeA[1]*sizeA[2])

################MPO################

MPON = torch.zeros((2,2,2,5,5), dtype=torch.float64)
MPOL = torch.zeros((2,2,2,5,5), dtype=torch.float64)
MPOR = torch.zeros((2,2,2,5,5), dtype=torch.float64)

# print(f"MPO1 :{MPO1}")
# MPON[0][0][1][0][0] = -ex_bx/2.0
# MPON[0][1][0][0][0] = -ex_bx/2.0
# MPON[1][0][1][0][0] = -ex_by/2.0
# MPON[1][1][0][0][0] = ex_by/2.0
MPON[0][0][0][0][4] = MPON[0][1][1][0][4] = 1.0
# MPON[0][0][0][4][4] = MPON[0][1][1][4][4] = 1.0
# MPON[0][0][1][0][1] = J/2.0

# MPON[0][1][0][0][1] = J/2.0
MPON[0][0][1][1][4] = 1.0
# MPON[0][1][0][1][4] = 1.0/2.0
# MPON[0][1][2][1][0] = MPON[0][1][2][5][5] = 1.0/np.sqrt(2.)
# MPON[0][2][1][1][0] = MPON[0][2][1][5][5] = 1.0/np.sqrt(2.)

# MPON[0][0][1][0][2] = J/2.0
# MPON[1][1][0][0][2] = -J/2.0
# MPON[1][0][1][2][4] = 1.0/2.0
MPON[0][1][0][2][4] = 1.0
# MPON[1][1][2][2][0] = MPON[1][1][2][6][6] = 1.0/np.sqrt(2.)
# MPON[1][2][1][2][0] = MPON[1][2][1][6][6] = -1.0/np.sqrt(2.)

# MPON[0][0][0][0][3] = J*delta / 2.0
# MPON[0][1][1][0][3] = -J*delta / 2.0
MPON[0][0][0][3][4] = 1.0 / 2.0
MPON[0][1][1][3][4] = -1.0 / 2.0
# MPON[0][2][2][3][0] = MPON[0][2][2][7][7] = -1.0

MPOL[0][0][0][0][0] = MPOL[0][1][1][0][0] = 1.0
MPOL[0][0][0][4][4] = MPOL[0][1][1][4][4] = 1.0

MPOL[0][0][1][1][0] = 1.0
# MPOL[0][1][0][1][0] = 1.0/2.0
# MPOL[0][1][2][1][0] = 1.0/np.sqrt(2.)
# MPOL[0][2][1][1][0] = 1.0/np.sqrt(2.)
# MPOL[1][0][1][2][0] = 1.0/2.0
##
MPOL[0][1][0][2][0] = 1.0
# MPOL[1][1][2][2][0] = 1.0/np.sqrt(2.)
# MPOL[1][2][1][2][0] = -1.0/np.sqrt(2.)
##
MPOL[0][0][0][3][0] = 1.0 / 2.0
MPOL[0][1][1][3][0] = -1.0 / 2.0
# MPOL[0][2][2][3][0] = -1.0
# MPOL[0][0][1][4][1] = J / 2.0
MPOL[0][1][0][4][1] = J / 2.0
# MPOL[0][1][2][4][1] = J / np.sqrt(2.)
# MPOL[0][2][1][4][1] = J / np.sqrt(2.)
MPOL[0][0][1][4][2] = J / 2.0
# MPOL[1][1][0][4][2] = -J / 2.0
# MPOL[1][1][2][4][2] = J / np.sqrt(2.)
# MPOL[1][2][1][4][2] = -J / np.sqrt(2.)
MPOL[0][0][0][4][3] = J * delta / 2.0
MPOL[0][1][1][4][3] = -J * delta / 2.0
MPOL[0][0][1][4][0] = -ex_bx / 2.0
MPOL[0][1][0][4][0] = -ex_bx / 2.0
MPOL[1][0][1][4][0] = ex_by / 2.0
MPOL[1][1][0][4][0] = -ex_by / 2.0

MPOR[0][0][0][0][0] = MPOR[0][1][1][0][0] = 1.0
MPOR[0][0][0][4][4] = MPOR[0][1][1][4][4] = 1.0
MPOR[0][0][1][1][0] = 1.0
# MPOR[0][1][0][1][0] = 1.0/2.0
# MPOR[0][1][2][1][0] = 1.0/np.sqrt(2.)
# MPOR[0][2][1][1][0] = 1.0/np.sqrt(2.)
# MPOR[1][0][1][2][0] = 1.0/2.0
MPOR[0][1][0][2][0] = 1.0
# MPOR[1][1][2][2][0] = 1.0/np.sqrt(2.)
# MPOR[1][2][1][2][0] = -1.0/np.sqrt(2.)
MPOR[0][0][0][3][0] = 1.0 / 2.0
MPOR[0][1][1][3][0] = -1.0 / 2.0
# MPOR[0][2][2][3][0] = -1.0
# MPOR[0][0][1][4][1] = J / 2.0
MPOR[0][1][0][4][1] = J / 2.0
# MPOR[0][1][2][4][1] = J / np.sqrt(2.)
# MPOR[0][2][1][4][1] = J / np.sqrt(2.)
MPOR[0][0][1][4][2] = J / 2.0
# MPOR[1][1][0][4][2] = -J / 2.0
# MPOR[1][1][2][4][2] = J / np.sqrt(2.)
# MPOR[1][2][1][4][2] = -J / np.sqrt(2.)
MPOR[0][0][0][4][3] = J * delta / 2.0
MPOR[0][1][1][4][3] = -J * delta / 2.0
MPOR[0][0][1][4][0] = -ex_bx / 2.0
MPOR[0][1][0][4][0] = -ex_bx / 2.0
MPOR[1][0][1][4][0] = -ex_by / 2.0
MPOR[1][1][0][4][0] = ex_by / 2.0

###########truncation of mpo###########
mpo_list = []
mpoi = contiguous_complex(einsum_complex('ijkl,mnlp->imjnkp', MPOL, MPOR))
mpoi = view_complex((sizeA[0],sizeA[0],5,5),mpoi)
mpo_list.append(mpoi)
mpoN = contiguous_complex(einsum_complex('ijkl,mnlp->imjnkp', MPOL, MPON))
mpoN = view_complex((sizeA[0],sizeA[0],5,5),mpoN)
mpo_list.append(mpoN)

print(f"mposize :{mpo_list[0].shape}")
##############take into account left and right environment##########
nor = contiguous_complex(einsum_complex('ijk,ijk->', AC, complex_conjugate(AC)))
# nor_AR = contiguous_complex(einsum_complex('ijk,ilm->jlkm', AR, complex_conjugate(AR)))
# temp = nor_AR.clone()
# for i in range(N):
#     nor_AR = contiguous_complex(einsum_complex('ijkl,klmn->ijmn', nor_AR, temp))
# nor = contiguous_complex(einsum_complex('ijkl,klmn->ijmn', nor, nor_AR))
# nor = view_complex((sizeA[1]**2,sizeA[2]**2),nor)
# nor = einsum_complex('i,il->l', El, nor)
# nor = einsum_complex('i,i->', nor, Er)
# print(nor.shape)
# nor = contiguous_complex(einsum_complex('i,i->',nor, temp_right))
print(f"nor:{nor}")
nor = np.sqrt(nor[0]**2+nor[1]**2)
print(f"nor:{nor}")

ene = einsum_complex('ijk,ilmn->jmknl', AC, mpo_list[1])
# tempN = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', tempN, complex_conjugate(A)))
ene = contiguous_complex(einsum_complex('jmkni,ijk->mn', ene, complex_conjugate(AC)))
ene = torch.trace(ene[0])
print(f"ene:{ene.shape}")
# ene = np.sqrt(ene[0]**2+ene[1]**2)
print(f"ene:{(ene/nor).item()}")
print(ene.item())

tempN = einsum_complex('ijk,ilmn->jmknl', C2, mpo_list[1])
tempN = einsum_complex('jmkni,ilo->mjlnko', tempN, complex_conjugate(C2))
tempN = view_complex((5,sizeA[1]**2,5,sizeB[2]**2),tempN)
ene = einsum_complex('j,ijkl->ikl', El, tempN)
ene = einsum_complex('ikl,l->ik', ene, Er)
ene = torch.trace(ene[0])
print(f"ene:{ene.shape}")
# ene = np.sqrt(ene[0]**2+ene[1]**2)
print(f"ene:{(ene/nor).item()}")
print(ene.item())
#########repeat##########
# temp1 = einsum_complex('ijk,ilmn->jmknl', AC, mpo_list[0])
# print(f"temp1size : {temp1.shape}")
# # temp1 = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', temp1, complex_conjugate(A)))
# temp1 = contiguous_complex(einsum_complex('jmkni,ijl->mnkl', temp1, complex_conjugate(AC)))
# # print(f"temp1size : {temp1.shape}")
# temp1 = view_complex((5,5*sizeA[2]**2),temp1)

# tempN = einsum_complex('ijk,ilmn->jmknl', AC, mpo_list[1])
# # tempN = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', tempN, complex_conjugate(A)))
# tempN = contiguous_complex(einsum_complex('jmkni,ilk->mjln', tempN, complex_conjugate(AC)))
# tempN = view_complex((5*sizeA[1]**2,5),tempN)
# # # tempi = einsum_complex('ijkl,imjnop->mnkolp', A, mpo_list[1])
# tempi = einsum_complex('ijk,ilmn->jmknl', AC, mpo_list[0])
# # tempi = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', tempi, complex_conjugate(A)))
# tempi = contiguous_complex(einsum_complex('jmkni,ilo->mjlnko', tempi, complex_conjugate(AC)))
# tempi = view_complex((5*sizeA[1]**2,5*sizeA[2]**2),tempi)
# ene = tempi.clone()
# # print(f"ene:{ene.shape}")
# for i in range(N-3):
#     ene = mm_complex(ene,tempi)
# # temp2 = mm_complex(tempi, tempi)
# ene = contiguous_complex(einsum_complex('ij,jl->il', temp1, ene))
# ene = contiguous_complex(einsum_complex('ij,jl->il', ene, tempN))
# # ene = contiguous_complex(einsum_complex('j,ijkl->ikl', El, temp1))
# # ene = contiguous_complex(einsum_complex('ikl,l->ik', ene, Er))
# ene = torch.trace(ene[0])
# print(f"ene:{ene.shape}")
# # ene = np.sqrt(ene[0]**2+ene[1]**2)
# print(f"ene:{(ene/nor).item()}")
# print(ene.item())
###########repeat############
# temp1 = einsum_complex('ijk,ilmn->jmknl', A, mpo_list[0])
# print(f"temp1size : {temp1.shape}")
# # temp1 = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', temp1, complex_conjugate(A)))
# temp1 = contiguous_complex(einsum_complex('jmkni,ilo->mjlnko', temp1, complex_conjugate(A)))
# # print(f"temp1size : {temp1.shape}")
# temp1 = view_complex((5,sizeA[1]**2,5*sizeA[2]**2),temp1)
# # tempN = einsum_complex('ijkl,imjnop->mnkolp', A, mpo_list[int(N/2-1)])
# tempN = einsum_complex('ijk,ilmn->jmknl', A, mpo_list[1])
# # tempN = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', tempN, complex_conjugate(A)))
# tempN = contiguous_complex(einsum_complex('jmkni,ilo->mjlkon', tempN, complex_conjugate(A)))
# tempN = view_complex((5*sizeA[1]**2,sizeA[2]**2,5),tempN)
# # tempi = einsum_complex('ijkl,imjnop->mnkolp', A, mpo_list[1])
# tempi = einsum_complex('ijk,ilmn->jmknl', A, mpo_list[0])
# # tempi = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', tempi, complex_conjugate(A)))
# tempi = contiguous_complex(einsum_complex('jmkni,ilo->jmlkno', tempi, complex_conjugate(A)))
# tempi = view_complex((5*sizeA[1]**2,5*sizeA[2]**2),tempi)
# ene = tempi.clone()
# # print(f"ene:{ene.shape}")
# for i in range(N-3):
#     ene = mm_complex(ene,tempi)
# # temp2 = mm_complex(tempi, tempi)
# ene = contiguous_complex(einsum_complex('ijk,kl->ijl', temp1, ene))
# ene = contiguous_complex(einsum_complex('ijk,klm->ijlm', ene, tempN))
# ene = contiguous_complex(einsum_complex('j,ijkl->ikl', El, ene))
# ene = contiguous_complex(einsum_complex('ikl,k->il', ene, Er))
# # for i in range(int(N/2/2-2)):
# #     ene = mm_complex(ene, temp2)
# ene = torch.trace(ene[0])
# print(f"ene:{ene.shape}")
# # ene = np.sqrt(ene[0]**2+ene[1]**2)
# print(f"ene:{(ene/nor).item()}")
# print(ene.item())

#######make the most left environment #########
# temp_left=contiguous_complex(einsum_complex('ijk,ijl->kl', A, complex_conjugate(A)))
# temp=contiguous_complex(einsum_complex('ijk,imn->jmkn', A, complex_conjugate(A)))
# nor = temp.clone()
# temp_right=contiguous_complex(einsum_complex('ijk,ilk->jl', A, complex_conjugate(A)))
# temp_right=view_complex((sizeA[1]**2),temp_right)
# # temp_right=view_complex((sizeA[1]**2),temp_right)
# for i in range(N-3):
#     nor = contiguous_complex(einsum_complex('ijkl,klmn->ijmn', nor, temp))
# nor = contiguous_complex(einsum_complex('ij,ijkl->kl', temp_left, nor))
# nor = view_complex((sizeA[2]**2),nor)
# # print(nor.shape)
# nor = contiguous_complex(einsum_complex('i,i->',nor, temp_right))
# nor = np.sqrt(nor[0]**2+nor[1]**2)
# print(f"nor:{nor}")

# temp1 = einsum_complex('ijk,ilmn->jmknl', A, mpo_list[0])
# print(f"temp1size : {temp1.shape}")
# # temp1 = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', temp1, complex_conjugate(A)))
# temp1 = contiguous_complex(einsum_complex('jmkni,ijl->mnkl', temp1, complex_conjugate(A)))
# # print(f"temp1size : {temp1.shape}")
# temp1 = view_complex((5,5*sizeA[2]**2),temp1)
# # tempN = einsum_complex('ijkl,imjnop->mnkolp', A, mpo_list[int(N/2-1)])
# tempN = einsum_complex('ijk,ilmn->jmknl', A, mpo_list[1])
# # tempN = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', tempN, complex_conjugate(A)))
# tempN = contiguous_complex(einsum_complex('jmkni,ilk->mjln', tempN, complex_conjugate(A)))
# tempN = view_complex((5*sizeA[1]**2,5),tempN)
# # tempi = einsum_complex('ijkl,imjnop->mnkolp', A, mpo_list[1])
# tempi = einsum_complex('ijk,ilmn->jmknl', A, mpo_list[0])
# # tempi = contiguous_complex(einsum_complex('ijklmn,ijop->klomnp', tempi, complex_conjugate(A)))
# tempi = contiguous_complex(einsum_complex('jmkni,ilo->mjlnko', tempi, complex_conjugate(A)))
# tempi = view_complex((5*sizeA[1]**2,5*sizeA[2]**2),tempi)
# ene = tempi.clone()
# print(f"ene:{ene.shape}")
# for i in range(N-3):
#     ene = mm_complex(ene,tempi)
# # temp2 = mm_complex(tempi, tempi)
# ene = mm_complex(temp1, ene)
# # for i in range(int(N/2/2-2)):
# #     ene = mm_complex(ene, temp2)
# ene = torch.trace(mm_complex(ene, tempN)[0])
# print(f"ene:{ene.shape}")
# print(f"ene:{(ene/N/nor/2).item()}")
# print(f"size {sizeA}")
# print(ene.item())