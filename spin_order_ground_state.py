import math
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
# from models import heisenbergs1
from complex_num.complex_operation import *
from optim.ad_optim import optimize_state
#from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)
import json
import multiprocessing as mp

tStart = time.time()
torch.pi = torch.tensor(np.pi, dtype=torch.float64)
J = 1.0
J2 = 0.0
chi = 20
N = 64
ex_bx = 0
# k = (0) * 2 * np.pi / N
# print (f"cosin {np.cos(k)}")
# print (np.sin(k))
# Kb = 1.380649e-23
Kb = 8.6171e-5
numofstate = 1
unit_len = 2
# model = heisenbergs1.Heisenbergs1(J, J2, N)
def lattice_to_site(coord):
    return (0)
state = read_ipeps("/home/23_takahashi/finite_tem/SrCo2V2O8/excitation_DQCP/ex-mps_n{}_chi{}_j1_y_h{}_state.json".format(N,chi,ex_bx), vertexToSite=lattice_to_site)
# energy_f=model.energy_2x2_1site
# print (f"state: {energy_f(state).item()}")
## set the distance, the far left site is regarded as 0
r_dis = 0

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


Vec = []
energy = []
state_whole = []
wave_number = []
############### use 20bond_ver2.json when calculating bond20 ##############
with open("/home/23_takahashi/finite_tem/SrCo2V2O8/excitation_DQCP/sorted_combined_{}bond_{}site_h{}.json".format(chi,N,ex_bx), "r") as f:
    file = json.load(f)
    for i in range(len(file)):
        state_element=[]
        energy.append(file[i]["ene"])
        for j in range(len(file[i]["state"][0])):
            real = np.float64(file[i]["state"][0][j])
            imag = np.float64(file[i]["state"][1][j])
            comp = complex(real, imag)
            state_element.append(comp)
        state_whole.append(state_element)

state_whole = torch.tensor(state_whole)

for i in range(numofstate):
    Vec.append(np.tensordot(state_whole[i].reshape(sizeA[0], sizeA[1], sizeA[2]),np.diag(1/C),([2],[0])))
    wave_number.append(file[i]["k"])
# for i in wave_number:
#     print(f"k: {i}")
################Partition function################
dic_ene = {}
cnt = 0
for ene in energy:
    dic_ene[cnt] = ene
    cnt+=1
# for i in range(numofstate):
#     print(f"ene:{dic_ene[i]}")


dict_pf = {}

def compute_partition_function(tem):
    if tem == 0:
        beta = np.float64("inf")
    else:
        beta = np.float64(1 / (Kb * tem))
    
    z = 0
    list_pf = []
    
    # 各状態に対して分配関数を計算
    for n in range(numofstate):
        z += np.exp(-beta * (dic_ene[n] - dic_ene[0] + 1e-9))  # 小さな定数を足してゼロ除算を防ぐ
        list_pf.append(z)
    
    return tem, list_pf  # 結果として温度と分配関数リストを返す

# 並列で分配関数を計算する関数
def parallel_compute_partition_functions():
    with mp.Pool(mp.cpu_count()) as pool:
        # 0Kから995Kまで5K刻みで分配関数を計算
        results = pool.map(compute_partition_function, range(0, 2005, 5))
    
    # 結果を辞書にまとめる
    dict_pf = {tem: list_pf for tem, list_pf in results}
    
    return dict_pf

# 並列計算の実行
dict_pf = parallel_compute_partition_functions()

# 結果の表示
print(f"pf : {dict_pf}")


################correlation################
sz_left = torch.zeros((2,4,4), dtype=torch.float64)
sz_right = torch.zeros((2,4,4), dtype=torch.float64)
sy = torch.zeros((2,4,4), dtype=torch.float64)
sz_left[0][0][0]= 1.0/2.0
sz_left[0][1][1]= 1.0/2.0
sz_left[0][2][2]= -1.0/2.0
sz_left[0][3][3]= -1.0/2.0
sz_right[0][0][0]= 1.0/2.0
sz_right[0][1][1]= -1.0/2.0
sz_right[0][2][2]= 1.0/2.0
sz_right[0][3][3]= -1.0/2.0




iden = torch.zeros((2,4,4), dtype=torch.float64)
iden[0][0][0]=1.0
iden[0][1][1]=1.0
iden[0][2][2]=1.0
iden[0][3][3]=1.0





# dict_cor = {}
norm_k=[]
final_result = []
cor_ful={}
final_cor = {}
# list_cor = [0 for i in range(numofstate)]


# nor_0 = view_complex((sizeA[1]**2, sizeA[2]**2),einsum_complex('ijk,imn->jmkn',A,complex_conjugate(A)))
# for i in range(N-1):
#     functional_DL = view_complex((sizeA[1]**2, sizeA[2]**2),einsum_complex('ijk,imn->jmkn',A,complex_conjugate(A)))
#     nor_0 = mm_complex(nor_0,functional_DL)
# norm_0 = torch.trace(nor_0[0]).item()

def compute_for_oprator(args):
    num, i = args
    exp = 0
    torch.autograd.set_detect_anomaly(True)
    

    # 各状態 `i` での計算処理
    lamb = torch.tensor(0.0, dtype=torch.float64).requires_grad_(True)
    lams = torch.tensor(0.0, dtype=torch.float64).requires_grad_(True)
    Vr = torch.as_tensor(Vec[i].real)
    Vi = torch.as_tensor(Vec[i].imag)
    k = wave_number[i]
    V = torch.stack((Vr, Vi), dim=0)
    norm_real = torch.tensor(0.0, dtype=torch.float64)
    norm_imag = torch.tensor(0.0, dtype=torch.float64)
    
    # 実際の計算ループ (N/unit_len) に対応
    for n in range(int(N/2/unit_len)):
        if n==0:
            for j in range(unit_len):
                Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
                if j==0:
                    Vk[0] = torch.tensor(np.cos(-k*(j))) * V[0] - torch.tensor(np.sin(-k*(j))) * V[1]
                    Vk[1] = torch.tensor(np.sin(-k*(j))) * V[0] + torch.tensor(np.cos(-k*(j))) * V[1]
                    temp_up = A 
                    temp_down = complex_conjugate(A)
                else:
                    Vk[0] = torch.tensor(np.cos(-k*(j))) * V[0] - torch.tensor(np.sin(-k*(j))) * V[1]
                    Vk[1] = torch.tensor(np.sin(-k*(j))) * V[0] + torch.tensor(np.cos(-k*(j))) * V[1]
                    func_up = A
                    func_down = complex_conjugate(A)
                    temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
                    temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
            DL = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
        elif n==int(N/2/unit_len)-1:
            for j in range(unit_len):
                Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
                if j==0:
                    Vk[0] = torch.tensor(np.cos(-k*(j+unit_len*n))) * V[0] - torch.tensor(np.sin(-k*(j+unit_len*n))) * V[1]
                    Vk[1] = torch.tensor(np.sin(-k*(j+unit_len*n))) * V[0] + torch.tensor(np.cos(-k*(j+unit_len*n))) * V[1]
                    temp_up = A
                    temp_down = complex_conjugate(A)
                else:
                    Vk[0] = torch.tensor(np.cos(-k*(j+unit_len*n))) * V[0] - torch.tensor(np.sin(-k*(j+unit_len*n))) * V[1]
                    Vk[1] = torch.tensor(np.sin(-k*(j+unit_len*n))) * V[0] + torch.tensor(np.cos(-k*(j+unit_len*n))) * V[1]
                    func_up = A
                    func_down = complex_conjugate(A)
                    temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
                    temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
            temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
            norm_real = einsum_complex('ijkl,klij',DL,temp)[0]
            norm_imag = einsum_complex('ijkl,klij',DL,temp)[1]
        else:
            for j in range(unit_len):
                Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
                if j==0:
                    Vk[0] = torch.tensor(np.cos(-k*(j+unit_len*n))) * V[0] - torch.tensor(np.sin(-k*(j+unit_len*n))) * V[1]
                    Vk[1] = torch.tensor(np.sin(-k*(j+unit_len*n))) * V[0] + torch.tensor(np.cos(-k*(j+unit_len*n))) * V[1]
                    temp_up = A 
                    temp_down = complex_conjugate(A)
                else:
                    Vk[0] = torch.tensor(np.cos(-k*(j+unit_len*n))) * V[0] - torch.tensor(np.sin(-k*(j+unit_len*n))) * V[1]
                    Vk[1] = torch.tensor(np.sin(-k*(j+unit_len*n))) * V[0] + torch.tensor(np.cos(-k*(j+unit_len*n))) * V[1]
                    func_up = A
                    func_down = complex_conjugate(A)
                    temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
                    temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
            temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
            DL = einsum_complex('ijkl,klmn->ijmn',DL,temp)

    # g = torch.autograd.grad(norm_real,lamb,create_graph=True)
    # g[0].backward(retain_graph=True)
    # norm_r = lams.grad.item()
    # g = torch.autograd.grad(norm_imag,lamb,create_graph=True)
    # g[0].backward(retain_graph=True)
    # norm_i = lams.grad.item() - norm_r
    # print(f"normimag : {norm_i}")
    # norm.append(devr)
    sz_list =[0.0,0.0]
    for site in range(2):
        lamb = torch.tensor(0.0,dtype=torch.float64).requires_grad_(True)
        lams = torch.tensor(0.0,dtype=torch.float64).requires_grad_(True)
        # lams = torch.tensor(0.0, dtype=torch.float64).requires_grad_(True)
        sz_real = torch.tensor(0.0,dtype=torch.float64)
        sz_imag = torch.tensor(0.0, dtype=torch.float64)
        # specr = torch.tensor(0.0, dtype=torch.float64)
        # speci = torch.tensor(0.0, dtype=torch.float64)
        # idenr = torch.eye(sizeA[0], dtype=torch.float64)
        # ideni = torch.zeros((sizeA[0],sizeA[0]), dtype=torch.float64)
        # iden = torch.stack((idenr, ideni), dim=0)

        for n in range(int(N/2/unit_len)):
            if n==0:
                for j in range(unit_len):
                    Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
                    # sk = torch.zeros((2,3,3), dtype=torch.float64)
                    if j==0:
                        Vk[0] = torch.tensor(np.cos(-k*(j)), dtype=torch.float64) * V[0] - torch.tensor(np.sin(-k*(j)), dtype=torch.float64) * V[1]
                        Vk[1] = torch.tensor(np.sin(-k*(j)), dtype=torch.float64) * V[0] + torch.tensor(np.cos(-k*(j)), dtype=torch.float64) * V[1]
                        temp_up = A
                        # sk[0] = torch.tensor(np.cos(k*(j)), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(k*(j)), dtype=torch.float64) * sz[1]
                        # sk[1] = torch.tensor(np.sin(k*(j)), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(k*(j)), dtype=torch.float64) * sz[1]
                        if site == 0:
                            temp_up = einsum_complex('ijk,il->ljk',temp_up,sz_left)
                        elif site ==1:
                            temp_up = einsum_complex('ijk,il->ljk',temp_up,sz_right)
                        temp_down = complex_conjugate(A)
                    else:
                        Vk[0] = torch.tensor(np.cos(-k*(j)), dtype=torch.float64) * V[0] - torch.tensor(np.sin(-k*(j)), dtype=torch.float64) * V[1]
                        Vk[1] = torch.tensor(np.sin(-k*(j)), dtype=torch.float64) * V[0] + torch.tensor(np.cos(-k*(j)), dtype=torch.float64) * V[1]
                        func_up = A
                        # sk[0] = torch.tensor(np.cos(k*(j)), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(k*(j)), dtype=torch.float64) * sz[1]
                        # sk[1] = torch.tensor(np.sin(k*(j)), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(k*(j)), dtype=torch.float64) * sz[1]
                        func_up = einsum_complex('ijk,il->ljk',func_up,iden)
                        func_down = complex_conjugate(A)
                        temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
                        temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
                DL = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
            elif n==int(N/2/unit_len)-1:
                for j in range(unit_len):
                    Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
                    # sk = torch.zeros((2,3,3), dtype=torch.float64)
                    if j==0:
                        Vk[0] = torch.tensor(np.cos(-k*(j+unit_len*n)), dtype=torch.float64) * V[0] - torch.tensor(np.sin(-k*(j+unit_len*n)), dtype=torch.float64) * V[1]
                        Vk[1] = torch.tensor(np.sin(-k*(j+unit_len*n)), dtype=torch.float64) * V[0] + torch.tensor(np.cos(-k*(j+unit_len*n)), dtype=torch.float64) * V[1]
                        temp_up = A 
                        # sk[0] = torch.tensor(np.cos(k*(j+unit_len*n)), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(k*(j+unit_len*n)), dtype=torch.float64) * sz[1]
                        # sk[1] = torch.tensor(np.sin(k*(j+unit_len*n)), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(k*(j+unit_len*n)), dtype=torch.float64) * sz[1]
                        temp_up = einsum_complex('ijk,il->ljk',temp_up,iden)
                        temp_down = complex_conjugate(A)
                    else:
                        Vk[0] = torch.tensor(np.cos(-k*(j+unit_len*n)), dtype=torch.float64) * V[0] - torch.tensor(np.sin(-k*(j+unit_len*n)), dtype=torch.float64) * V[1]
                        Vk[1] = torch.tensor(np.sin(-k*(j+unit_len*n)), dtype=torch.float64) * V[0] + torch.tensor(np.cos(-k*(j+unit_len*n)), dtype=torch.float64) * V[1]
                        func_up = A
                        # sk[0] = torch.tensor(np.cos(k*(j+unit_len*n)), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(k*(j+unit_len*n)), dtype=torch.float64) * sz[1]
                        # sk[1] = torch.tensor(np.sin(k*(j+unit_len*n)), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(k*(j+unit_len*n)), dtype=torch.float64) * sz[1]
                        func_up = einsum_complex('ijk,il->ljk',func_up,iden)
                        func_down = complex_conjugate(A)
                        temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
                        temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
                temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
                DL = einsum_complex('ijkl,klij',DL,temp)
                sz_real = DL[0]
                sz_imag = DL[1]
                # specr = DL[0]
                # speci = DL[1]
            else:
                for j in range(unit_len):
                    Vk = torch.zeros((2,sizeA[0],sizeA[1],sizeA[2]), dtype=torch.float64)
                    # sk = torch.zeros((2,3,3), dtype=torch.float64)
                    if j==0:
                        Vk[0] = torch.tensor(np.cos(-k*(j+unit_len*n)), dtype=torch.float64) * V[0] - torch.tensor(np.sin(-k*(j+unit_len*n)), dtype=torch.float64) * V[1]
                        Vk[1] = torch.tensor(np.sin(-k*(j+unit_len*n)), dtype=torch.float64) * V[0] + torch.tensor(np.cos(-k*(j+unit_len*n)), dtype=torch.float64) * V[1]
                        temp_up = A
                        # sk[0] = torch.tensor(np.cos(k*(j+unit_len*n)), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(k*(j+unit_len*n)), dtype=torch.float64) * sz[1]
                        # sk[1] = torch.tensor(np.sin(k*(j+unit_len*n)), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(k*(j+unit_len*n)), dtype=torch.float64) * sz[1]
                        temp_up = einsum_complex('ijk,il->ljk',temp_up,iden)
                        temp_down = complex_conjugate(A)
                    else:
                        Vk[0] = torch.tensor(np.cos(-k*(j+unit_len*n)), dtype=torch.float64) * V[0] - torch.tensor(np.sin(-k*(j+unit_len*n)), dtype=torch.float64) * V[1]
                        Vk[1] = torch.tensor(np.sin(-k*(j+unit_len*n)), dtype=torch.float64) * V[0] + torch.tensor(np.cos(-k*(j+unit_len*n)), dtype=torch.float64) * V[1]
                        func_up = A
                        # sk[0] = torch.tensor(np.cos(k*(j+unit_len*n)), dtype=torch.float64) * sz[0] - torch.tensor(np.sin(k*(j+unit_len*n)), dtype=torch.float64) * sz[1]
                        # sk[1] = torch.tensor(np.sin(k*(j+unit_len*n)), dtype=torch.float64) * sz[0] + torch.tensor(np.cos(k*(j+unit_len*n)), dtype=torch.float64) * sz[1]
                        func_up = einsum_complex('ijk,il->ljk',func_up,iden)
                        func_down = complex_conjugate(A)
                        temp_up = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_up,func_up))
                        temp_down = view_complex((sizeA[0]**(j+1),sizeA[1],sizeA[2]),einsum_complex('ijk,lkn->iljn',temp_down,func_down))
                temp = einsum_complex('ijk,imn->jmkn',temp_up,temp_down)
                DL = einsum_complex('ijkl,klmn->ijmn',DL,temp)
            
        # g = torch.autograd.grad(sz_real,lamb,create_graph=True)
        # g[0].backward(retain_graph=True)
        # devr = lams.grad.item()
        # g = torch.autograd.grad(sz_imag,lamb,create_graph=True)
        # g[0].backward(retain_graph=True)
        # devi = lams.grad.item() - devr
        # print(f"num_state:{i}")
        # print(f"temp:{tem}")
        # print(f"cor_real:{devr}")
        # print(f"cor_imag:{devi}")
        # print(f"norm : {norm_r}")
        # devr /= (norm_r)
        sz_list[site] = sz_real/norm_real
    print(f"sz:{sz_list}")
    sz_value = (sz_list[0] - sz_list[1]) /2 
    
    return num, i, sz_value
    
# 並列で温度ごとの計算を実行
def parallel_compute():
    # 温度と状態を全ての組み合わせで並列処理
    # num=0
    # args = [(num, i) for i in range(numofstate)]
    args = [(num, i) for num in range(1) for i in range(numofstate)]
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(compute_for_oprator, args)
    dict_cor = {}
    # 結果を整理
    for num, i, devr in results:
        if num not in dict_cor:
            dict_cor[num] = [0 for _ in range(numofstate)]
        dict_cor[num][i] = (devr)
    
    return dict_cor

     
def compute_for_temperature(args):
    tem, i = args
    list_cor = [0] * numofstate  # 各プロセスでローカルな list_cor を作成
    
    if tem == 0:
        list_cor[0] = final_result[0]
        return tem, list_cor
    else:
        beta = np.float(1 / (Kb * tem))
        numerator = 0
        for i in range(numofstate):
            numerator += np.exp(-beta * (dic_ene[i] - dic_ene[0] + 1e-9)) * final_result[i]
            list_cor[i] = numerator / dict_pf[tem][i]
        
        return tem, list_cor


def parallel_compute_for_temperature():
    cor_temp = {}
    
    # for tem in range(0, 2005, 5):
    #     list_cor = [0] * numofstate  # 各温度ごとに `list_cor` を初期化
        
    #     args = [(tem, i, list_cor) for i in range(numofstate)]  # 前の値を次の計算に引き継ぐ
        
    #     with mp.Pool(mp.cpu_count()) as pool:
    #         results = pool.map(compute_for_temperature, args)
        
        # `list_cor` を更新して結果を統合
        # for tem, i, updated_list_cor in results:
        #     list_cor = updated_list_cor  # 更新された `list_cor` を使う
        
        # cor_temp[tem] = list_cor  # 最終的な `list_cor` を保存
    
    # return cor_temp
    # args = [(tem, i) for tem in range(0,2005,5) for i in range(numofstate)]
    args = [(tem, i) for tem in range(1) for i in range(numofstate)]
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(compute_for_temperature, args)
    # cor_temp = {}
    # for tem, list_cor in results:
    #     if tem not in dict_cor:
    #         dict_cor[tem] = list_cor
    # 結果を辞書にまとめる
    for tem, list_cor_value in results:
        if tem not in cor_temp:
            cor_temp[tem] = [0] * numofstate  # 初期化
        cor_temp[tem] = list_cor_value   # 各 `i` 番目の値を保存
    
    return cor_temp
    # cor_temp = {tem: list_cor for tem, list_cor in results}
    # return cor_temp


# 並列計算の実行
cor = parallel_compute()
# print(f"cor:{cor}")
print(f"cor:{(cor)}")
final_result = [sum(x) for x in zip(*cor.values())]

print(f"res: {(final_result)}")



# 並列計算の実行
final_cor = parallel_compute_for_temperature()
cor_ful[r_dis] = final_cor

# 結果の表示
# print(final_cor)


# with open('mps_temp_sz_{}bond_h{}_Sr.json'.format(chi,ex_bx), 'w') as f:
#     json.dump(cor_ful, f, indent=1)


