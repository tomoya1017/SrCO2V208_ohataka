import torch
import groups.S2 as S2
import config as cfg
import numpy as np
from math import sqrt
from complex_num.complex_operation import *
import itertools

class DQCP():
    def __init__(self, J, delta, gx, bohr, h, N, global_args=cfg.global_args):
        r"""
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param global_args: global configuration
        :type j1: float
        :type j2: float
        :type global_args: GLOBALARGS

        Build Spin-1/2 :math:`J_1-J_2` Hamiltonian

        .. math:: H = J_1\sum_{<i,j>} h2_{ij} + J_2\sum_{<<i,j>>} h2_{ij}

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), and the second sum runs over 
        pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`)::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
                :  :  :  :

        where

        * :math:`h2_{ij} = \mathbf{S_i}.\mathbf{S_j}` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.J=J
        self.delta=delta
        self.gx = gx
        self.bohr=bohr
        self.h=h
        self.hy=0.29 * self.h
        # self.hz=0.14 * self.h
        self.N=N
        
        self.SSx, self.SSy, self.SSz, self.Sx, self.Sy, self.Sz, self.SS = self.get_h()
        self.obs_ops= self.get_obs_ops()

    def get_h(self):
        s2 = S2.S2(self.phys_dim, dtype=self.dtype, device=self.device)
        expr_kron = 'ij,ab->iajb'
        SSx= einsum_complex(expr_kron,s2.SX(),s2.SX())
        SSx= contiguous_complex(SSx)
        SSy= einsum_complex(expr_kron,s2.SY(),s2.SY())
        SSy= contiguous_complex(SSy)
        SSz= einsum_complex(expr_kron,s2.SZ(),s2.SZ()) 
        SSz= contiguous_complex(SSz)
        SS= einsum_complex(expr_kron,s2.SX(),s2.SX()) + einsum_complex(expr_kron,s2.SY(),s2.SY()) + einsum_complex(expr_kron,s2.SZ(),s2.SZ()) 
        SS= contiguous_complex(SS)
        
        # ここも直す
        return SSx, SSy, SSz, s2.SX(), s2.SY(), s2.SZ(), SS

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = S2.S2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sx"]= s2.SX()
        obs_ops["sy"]= s2.SY()
        return obs_ops

    # def energy_2x2_1site(self,state):
        mps = state.sites[(0)]
        ######### caluculate norm in order ##########
        # dict_norm={}
        # first_norm = einsum_complex(("ijk,imn->jmkn",mps,complex_conjugate(mps)))
        # small_norm = first_norm
        # dict_norm[0] = first_norm
        # for i in range(1,self.N):
        #     # small_norm = einsum_complex(("ijk,imn->jmkn",m,complex_conjugate(mps)))
        #     small_norm = einsum_complex(("ijkl,klmn->ijmn",small_norm,first_norm))
        #     dict_norm[i] = small_norm
        
        norm1 = torch.eye(size_complex(mps)[2]**2,dtype=self.dtype,device=self.device)
        norm2 = torch.zeros((size_complex(mps)[2]**2,size_complex(mps)[2]**2),dtype=self.dtype,device=self.device)
        norm = torch.stack((norm1, norm2), dim=0)
        G_list_contract = einsum_complex('ijk,imn->jmkn',mps,complex_conjugate(mps))#.reshape(G_list[j].size()[1]**2,G_list[j].size()[2]**2))
        G_list_contract = view_complex((size_complex(mps)[1]**2,size_complex(mps)[2]**2), G_list_contract)
        for j in range(0,self.N):
            norm = mm_complex(norm,G_list_contract)
        norm = trace_complex(norm)
        theta = einsum_complex('ijk,lkm->ijlm',mps,mps)
        # ここを直す
        theta_O = einsum_complex('ijkl,kmln->imjn',(self.J*self.SSx+self.J*self.SSy+self.J*self.delta*self.SSz),theta)
        E = einsum_complex('ijkl,imkn->jmln',theta_O,complex_conjugate(theta))#.reshape(G_list[(j)%size].size()[1]**2,G_list[(j+1)%size].size()[2]**2)
        E = view_complex((size_complex(mps)[1]**2,size_complex(mps)[2]**2), E)
        for i in range(1,self.N-1):
            E = mm_complex(E,G_list_contract)
        E_nn = trace_complex(E)/norm
        ここを直す
        for i in range(self.N):
            if (i==0):
                Ham = einsum_complex('ijk,il->ljk',mps,(self.bohr*self.gx*self.h*self.Sx + self.bohr*self.gx*self.hy*(-1**i)*self.Sy + self.bohr*self.gx*self.hz*self.Sz*np.cos(np.pi(2*i-1)/4)))
                Ham = einsum_complex("ljk,lml->jmkl",Ham,complex_conjugate(mps))
                bear_mps = einsum_complex
        theta_O = einsum_complex('ijkl,kmn->imnjl',(self.bohr*self.gx*self.h*self.Sx + self.bohr*self.gx*self.hy*self.Sy + self.bohr*self.gx*self.hz*self.Sz),mps)
        theta_O = einsum_complex('imnjl,lab->imnjab',theta_O,mps)
        theta_O = einsum_complex('imnjab,kna->imkjb',theta_O,mps)
        E = einsum_complex('imkjb,ipq->pqmkjb',theta_O,complex_conjugate(mps))
        E = einsum_complex('pqmkjb,kqja->pmab',E,complex_conjugate(theta))
        E = view_complex((size_complex(mps)[1]**2,size_complex(mps)[2]**2), E)
        for i in range(2,self.N-1):
            E = mm_complex(E,G_list_contract)
        E_nnn = trace_complex(E)/norm

        #theta_O = einsum_complex('ji,ikl->jkl',2.0*self.SZ,mps)
        #E = einsum_complex('ijk,imn->jmkn',theta_O,complex_conjugate(mps))#.reshape(G_list[j].size()[1]**2,G_list[j].size()[2]**2)
        #E = view_complex((size_complex(mps)[1]**2,size_complex(mps)[2]**2), E)
        #for i in range(1,self.N):
        #    E = mm_complex(E,G_list_contract)
        #E_s = trace_complex(E)/norm

        return E_nn+E_nnn

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

        ############# next nearest exchange ##########
        # theta = einsum_complex('ijk,lkm->ijlm',mps2,mps1)
        # theta_O = einsum_complex('ijkl,kmn->imnjl',(self.Kx*self.SSx+self.Kz*self.SSz),mps1)
        # theta_O = einsum_complex('imnjl,lab->imnjab',theta_O,mps1)
        # theta_O = einsum_complex('imnjab,kna->imkjb',theta_O,mps2)
        # E1 = einsum_complex('imkjb,ipq->pqmkjb',theta_O,complex_conjugate(mps1))
        # E1 = einsum_complex('pqmkjb,kqja->pmab',E1,complex_conjugate(theta))
        # E1 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), E1)
        # for i in range(2,self.N-1):
        #     if i % 2 == 1:
        #         E1 = mm_complex(E1,G_list_contract1)
        #     else:
        #         E1 = mm_complex(E1,G_list_contract2)
        # theta = einsum_complex('ijk,lkm->ijlm',mps1,mps2)
        # theta_O = einsum_complex('ijkl,kmn->imnjl',(self.Kx*self.SSx+self.Kz*self.SSz),mps2)
        # theta_O = einsum_complex('imnjl,lab->imnjab',theta_O,mps2)
        # theta_O = einsum_complex('imnjab,kna->imkjb',theta_O,mps1)
        # E2 = einsum_complex('imkjb,ipq->pqmkjb',theta_O,complex_conjugate(mps2))
        # E2 = einsum_complex('pqmkjb,kqja->pmab',E2,complex_conjugate(theta))
        # E2 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), E2)
        # for i in range(2,self.N-1):
        #     if i % 2 == 0:
        #         E2 = mm_complex(E2,G_list_contract1)
        #     else:
        #         E2 = mm_complex(E2,G_list_contract2)
        # E_nnn = (trace_complex(E1)+trace_complex(E2))/norm/2.


        ########## onsite operator s^x ###########

        # theta_Ope_x1 = einsum_complex("ijk,il->ljk",mps1,self.bohr*self.gx*self.h*self.Sx)
        theta_Ope_x1 = einsum_complex("ijk,il->ljk",mps1,self.h*self.Sx)
        theta_Ope_x1 = einsum_complex("ijk,ilm->jlkm",theta_Ope_x1,complex_conjugate(mps1))
        Ex1 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), theta_Ope_x1)
        # theta_Ope_x2 = einsum_complex("ijk,il->ljk",mps2,self.bohr*self.gx*self.h*self.Sx)
        theta_Ope_x2 = einsum_complex("ijk,il->ljk",mps2,self.h*self.Sx)
        theta_Ope_x2 = einsum_complex("ijk,ilm->jlkm",theta_Ope_x2,complex_conjugate(mps2))
        Ex2 = view_complex((size_complex(mps2)[1]**2,size_complex(mps2)[2]**2), theta_Ope_x2)
        for i in range(1,self.N):
            if i % 2 == 0:
                Ex1 = mm_complex(Ex1,G_list_contract1)
            else:
                Ex1 = mm_complex(Ex1,G_list_contract2)
        # Ex1*=self.N
        for i in range(1,self.N):
            if i % 2 == 1:
                Ex2 = mm_complex(Ex2,G_list_contract1)
            else:
                Ex2 = mm_complex(Ex2,G_list_contract2)
        # Ex2*=self.N
        E_x = (trace_complex(Ex1)+trace_complex(Ex2))/norm/2

        ########## onsite operator s^y ###########
        # theta_Ope_y1 = einsum_complex("ijk,il->ljk",mps1,self.bohr*self.gx*self.hy*self.Sy)
        theta_Ope_y1 = einsum_complex("ijk,il->ljk",mps1,self.hy*self.Sy)
        theta_Ope_y1 = einsum_complex("ijk,ilm->jlkm",theta_Ope_y1,complex_conjugate(mps1))
        Ey1 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), theta_Ope_y1)
        # theta_Ope_y2 = einsum_complex("ijk,il->ljk",mps2,-self.bohr*self.gx*self.hy*self.Sy)
        theta_Ope_y2 = einsum_complex("ijk,il->ljk",mps2,-self.hy*self.Sy)
        theta_Ope_y2 = einsum_complex("ijk,ilm->jlkm",theta_Ope_y2,complex_conjugate(mps2))
        Ey2 = view_complex((size_complex(mps2)[1]**2,size_complex(mps2)[2]**2), theta_Ope_y2)
        for i in range(1,self.N):
            if i % 2 == 0:
                Ey1 = mm_complex(Ey1,G_list_contract1)
            else:
                Ey1 = mm_complex(Ey1,G_list_contract2)
        # Ey1*=self.N
        for i in range(1,self.N):
            if i % 2 == 1:
                Ey2 = mm_complex(Ey2,G_list_contract1)
            else:
                Ey2 = mm_complex(Ey2,G_list_contract2)
        # Ey2*=self.N
        E_y = (trace_complex(Ey1)+trace_complex(Ey2))/norm/2

        


        # for i in range(self.N):

        #     if (i==0):
        #         Ham = einsum_complex('ijk,il->ljk',mps,(self.bohr*self.gx*self.h*self.Sx + self.bohr*self.gx*self.hy*(-1**i)*self.Sy + self.bohr*self.gx*self.hz*self.Sz*np.cos(np.pi(2*i-1)/4)))
        #         Ham = einsum_complex("ljk,lml->jmkl",Ham,complex_conjugate(mps))
        #         bear_mps = einsum_complex
        # theta = einsum_complex('ijk,lkm->ijlm',mps2,mps1)
        # theta_O = einsum_complex('ijkl,kmn->imnjl',(self.Kx*self.SSx+self.Kz*self.SSz),mps1)
        # theta_O = einsum_complex('imnjl,lab->imnjab',theta_O,mps1)
        # theta_O = einsum_complex('imnjab,kna->imkjb',theta_O,mps2)
        # E1 = einsum_complex('imkjb,ipq->pqmkjb',theta_O,complex_conjugate(mps1))
        # E1 = einsum_complex('pqmkjb,kqja->pmab',E1,complex_conjugate(theta))
        # E1 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), E1)
        # for i in range(2,self.N-1):
        #     if i % 2 == 1:
        #         E1 = mm_complex(E1,G_list_contract1)
        #     else:
        #         E1 = mm_complex(E1,G_list_contract2)
        # theta = einsum_complex('ijk,lkm->ijlm',mps1,mps2)
        # theta_O = einsum_complex('ijkl,kmn->imnjl',(self.Kx*self.SSx+self.Kz*self.SSz),mps2)
        # theta_O = einsum_complex('imnjl,lab->imnjab',theta_O,mps2)
        # theta_O = einsum_complex('imnjab,kna->imkjb',theta_O,mps1)
        # E2 = einsum_complex('imkjb,ipq->pqmkjb',theta_O,complex_conjugate(mps2))
        # E2 = einsum_complex('pqmkjb,kqja->pmab',E2,complex_conjugate(theta))
        # E2 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), E2)
        # for i in range(2,self.N-1):
        #     if i % 2 == 0:
        #         E2 = mm_complex(E2,G_list_contract1)
        #     else:
        #         E2 = mm_complex(E2,G_list_contract2)
        # E_nnn = (trace_complex(E1)+trace_complex(E2))/norm/2.

        #theta_O = einsum_complex('ji,ikl->jkl',2.0*self.SZ,mps)
        #E = einsum_complex('ijk,imn->jmkn',theta_O,complex_conjugate(mps))#.reshape(G_list[j].size()[1]**2,G_list[j].size()[2]**2)
        #E = view_complex((size_complex(mps)[1]**2,size_complex(mps)[2]**2), E)
        #for i in range(1,self.N):
        #    E = mm_complex(E,G_list_contract)
        #E_s = trace_complex(E)/norm

        return E_nn-(E_x+E_y)
    

    ############## spin order s^z for Ground_state #################
    def energy_2x2_2site_sz_gs(self,A1,A2):
        s2 = S2.S2(self.phys_dim, dtype=self.dtype, device=self.device)
        mps1 = A1
        mps2 = A2
        
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
        theta_Ope_z1 = einsum_complex("ijk,il->ljk",mps1,s2.SZ())
        theta_Ope_z1 = einsum_complex("ijk,ilm->jlkm",theta_Ope_z1,complex_conjugate(mps1))
        order_1 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), theta_Ope_z1)
        theta_Ope_z2 = einsum_complex("ijk,il->ljk",mps2,s2.SZ())
        theta_Ope_z2 = einsum_complex("ijk,ilm->jlkm",theta_Ope_z2,complex_conjugate(mps2))
        order_2 = view_complex((size_complex(mps2)[1]**2,size_complex(mps2)[2]**2), theta_Ope_z2)
        for i in range(1,self.N):
            if i % 2 == 0:
                order_1 = mm_complex(order_1,G_list_contract1)
            else:
                order_1 = mm_complex(order_1,G_list_contract2)
        # Ex1*=self.N
        for i in range(1,self.N):
            if i % 2 == 1:
                order_2 = mm_complex(order_2,G_list_contract1)
            else:
                order_2 = mm_complex(order_2,G_list_contract2)
        # Ex2*=self.N
        order_z = (trace_complex(order_1)-trace_complex(order_2))/norm/2
        print(f"S_A^z :{trace_complex(order_1)/norm}")
        print(f"S_B^z :{trace_complex(order_2)/norm}")

        return order_z
    
    def energy_2x2_2site_sx_gs(self,A1,A2):
        s2 = S2.S2(self.phys_dim, dtype=self.dtype, device=self.device)
        mps1 = A1
        mps2 = A2
        
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
        theta_Ope_x1 = einsum_complex("ijk,il->ljk",mps1,s2.SX().clone().detach())
        theta_Ope_x1 = einsum_complex("ijk,ilm->jlkm",theta_Ope_x1,complex_conjugate(mps1))
        order_1 = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), theta_Ope_x1)
        theta_Ope_x2 = einsum_complex("ijk,il->ljk",mps2,s2.SX().clone().detach())
        theta_Ope_x2 = einsum_complex("ijk,ilm->jlkm",theta_Ope_x2,complex_conjugate(mps2))
        order_2 = view_complex((size_complex(mps2)[1]**2,size_complex(mps2)[2]**2), theta_Ope_x2)
        for i in range(1,self.N):
            if i % 2 == 0:
                order_1 = mm_complex(order_1,G_list_contract1)
            else:
                order_1 = mm_complex(order_1,G_list_contract2)
        # Ex1*=self.N
        for i in range(1,self.N):
            if i % 2 == 1:
                order_2 = mm_complex(order_2,G_list_contract1)
            else:
                order_2 = mm_complex(order_2,G_list_contract2)
        # Ex2*=self.N
        order_x = (trace_complex(order_1)+trace_complex(order_2))/norm/2

        return order_x

    def eval_obs(self,state):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. average magnetization over the unit cell,
            2. magnetization for each site in the unit cell
            3. :math:`\langle S^z \rangle,\ \langle S^+ \rangle,\ \langle S^- \rangle` 
               for each site in the unit cell

        where the on-site magnetization is defined as
        
        .. math::
            
            \begin{align*}
            m &= \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }
            =\sqrt{\langle S^z \rangle^2+1/4(\langle S^+ \rangle+\langle S^- 
            \rangle)^2 -1/4(\langle S^+\rangle-\langle S^-\rangle)^2} \\
              &=\sqrt{\langle S^z \rangle^2 + 1/2\langle S^+ \rangle \langle S^- \rangle)}
            \end{align*}

        Usual spin components can be obtained through the following relations
        
        .. math::
            
            \begin{align*}
            S^+ &=S^x+iS^y               & S^x &= 1/2(S^+ + S^-)\\
            S^- &=S^x-iS^y\ \Rightarrow\ & S^y &=-i/2(S^+ - S^-)
            \end{align*}
        """
        # TODO optimize/unify ?
        # expect "list" of (observable label, value) pairs ?
        obs= dict({"avg_sz": 0.})
        s2 = S2.S2(self.phys_dim, dtype=self.dtype, device=self.device)
        with torch.no_grad():
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
            theta_SS = einsum_complex('ijkl,kmln->imjn',self.SS,theta)
            avg_SS = einsum_complex('ijkl,imkn->jmln',theta_SS,complex_conjugate(theta))#.reshape(G_list[(j)%size].size()[1]**2,G_list[(j+1)%size].size()[2]**2)
            avg_SS = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), avg_SS)
            for i in range(1,self.N-1):
                if i % 2 == 1:
                    avg_SS = mm_complex(avg_SS,G_list_contract1)
                else:
                    avg_SS = mm_complex(avg_SS,G_list_contract2)
            obs["avg_SS1"] = trace_complex(avg_SS)/norm

            theta = einsum_complex('ijk,lkm->ijlm',mps2,mps1)
            theta_SS = einsum_complex('ijkl,kmln->imjn',self.SS,theta)
            avg_SS = einsum_complex('ijkl,imkn->jmln',theta_SS,complex_conjugate(theta))#.reshape(G_list[(j)%size].size()[1]**2,G_list[(j+1)%size].size()[2]**2)
            avg_SS = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), avg_SS)
            for i in range(1,self.N-1):
                if i % 2 == 0:
                    avg_SS = mm_complex(avg_SS,G_list_contract1)
                else:
                    avg_SS = mm_complex(avg_SS,G_list_contract2)
            obs["avg_SS2"] = trace_complex(avg_SS)/norm

            theta = einsum_complex('ijk,lkm->ijlm',mps1,mps2)
            theta_SSx = einsum_complex('ijkl,kmln->imjn',self.SSx,theta)
            avg_SSx = einsum_complex('ijkl,imkn->jmln',theta_SSx,complex_conjugate(theta))#.reshape(G_list[(j)%size].size()[1]**2,G_list[(j+1)%size].size()[2]**2)
            avg_SSx = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), avg_SSx)
            for i in range(1,self.N-1):
                if i % 2 == 1:
                    avg_SSx = mm_complex(avg_SSx,G_list_contract1)
                else:
                    avg_SSx = mm_complex(avg_SSx,G_list_contract2)
            obs["avg_SSx1"] = trace_complex(avg_SSx)/norm

            theta = einsum_complex('ijk,lkm->ijlm',mps2,mps1)
            theta_SSx = einsum_complex('ijkl,kmln->imjn',self.SSx,theta)
            avg_SSx = einsum_complex('ijkl,imkn->jmln',theta_SSx,complex_conjugate(theta))#.reshape(G_list[(j)%size].size()[1]**2,G_list[(j+1)%size].size()[2]**2)
            avg_SSx = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), avg_SSx)
            for i in range(1,self.N-1):
                if i % 2 == 0:
                    avg_SSx = mm_complex(avg_SSx,G_list_contract1)
                else:
                    avg_SSx = mm_complex(avg_SSx,G_list_contract2)
            obs["avg_SSx2"] = trace_complex(avg_SSx)/norm

            #theta_SSy = einsum_complex('ijkl,kmln->imjn',self.SSy,theta)
            #avg_SSy = einsum_complex('ijkl,imkn->jmln',theta_SSy,complex_conjugate(theta))#.reshape(G_list[(j)%size].size()[1]**2,G_list[(j+1)%size].size()[2]**2)
            #avg_SSy = view_complex((size_complex(mps)[1]**2,size_complex(mps)[2]**2), avg_SSy)
            #for i in range(1,self.N-1):
            #    avg_SSy = mm_complex(avg_SSy,G_list_contract)
            #obs["avg_SSy"] = trace_complex(avg_SSy)/norm

            theta = einsum_complex('ijk,lkm->ijlm',mps1,mps2)
            theta_SSz = einsum_complex('ijkl,kmln->imjn',self.SSz,theta)
            avg_SSz = einsum_complex('ijkl,imkn->jmln',theta_SSz,complex_conjugate(theta))#.reshape(G_list[(j)%size].size()[1]**2,G_list[(j+1)%size].size()[2]**2)
            avg_SSz = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), avg_SSz)
            for i in range(1,self.N-1):
                if i % 2 == 1:
                    avg_SSz = mm_complex(avg_SSz,G_list_contract1)
                else:
                    avg_SSz = mm_complex(avg_SSz,G_list_contract2)
            obs["avg_SSz1"] = trace_complex(avg_SSz)/norm

            theta = einsum_complex('ijk,lkm->ijlm',mps2,mps1)
            theta_SSz = einsum_complex('ijkl,kmln->imjn',self.SSz,theta)
            avg_SSz = einsum_complex('ijkl,imkn->jmln',theta_SSz,complex_conjugate(theta))#.reshape(G_list[(j)%size].size()[1]**2,G_list[(j+1)%size].size()[2]**2)
            avg_SSz = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), avg_SSz)
            for i in range(1,self.N-1):
                if i % 2 == 0:
                    avg_SSz = mm_complex(avg_SSz,G_list_contract1)
                else:
                    avg_SSz = mm_complex(avg_SSz,G_list_contract2)
            obs["avg_SSz2"] = trace_complex(avg_SSz)/norm

            theta_SZ = einsum_complex('ji,ikl->jkl',s2.SZ(),mps1)
            avg_SZ = einsum_complex('ijk,imn->jmkn',theta_SZ,complex_conjugate(mps1))#.reshape(G_list[j].size()[1]**2,G_list[j].size()[2]**2)
            avg_SZ = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), avg_SZ)
            for i in range(1,self.N):
                if i % 2 == 0:
                    avg_SZ = mm_complex(avg_SZ,G_list_contract1)
                else:
                    avg_SZ = mm_complex(avg_SZ,G_list_contract2)
            obs["avg_sz1"] = trace_complex(avg_SZ)/norm

            theta_SZ = einsum_complex('ji,ikl->jkl',s2.SZ(),mps2)
            avg_SZ = einsum_complex('ijk,imn->jmkn',theta_SZ,complex_conjugate(mps2))#.reshape(G_list[j].size()[1]**2,G_list[j].size()[2]**2)
            avg_SZ = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), avg_SZ)
            for i in range(1,self.N):
                if i % 2 == 1:
                    avg_SZ = mm_complex(avg_SZ,G_list_contract1)
                else:
                    avg_SZ = mm_complex(avg_SZ,G_list_contract2)
            obs["avg_sz2"] = trace_complex(avg_SZ)/norm

            theta_SX = einsum_complex('ji,ikl->jkl',s2.SX(),mps1)
            avg_SX = einsum_complex('ijk,imn->jmkn',theta_SX,complex_conjugate(mps1))#.reshape(G_list[j].size()[1]**2,G_list[j].size()[2]**2)
            avg_SX = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), avg_SX)
            for i in range(1,self.N):
                if i % 2 == 0:
                    avg_SX = mm_complex(avg_SX,G_list_contract1)
                else:
                    avg_SX = mm_complex(avg_SX,G_list_contract2)
            obs["avg_sx1"] = trace_complex(avg_SX)/norm

            theta_SX = einsum_complex('ji,ikl->jkl',s2.SX(),mps2)
            avg_SX = einsum_complex('ijk,imn->jmkn',theta_SX,complex_conjugate(mps2))#.reshape(G_list[j].size()[1]**2,G_list[j].size()[2]**2)
            avg_SX = view_complex((size_complex(mps1)[1]**2,size_complex(mps1)[2]**2), avg_SX)
            for i in range(1,self.N):
                if i % 2 == 1:
                    avg_SX = mm_complex(avg_SX,G_list_contract1)
                else:
                    avg_SX = mm_complex(avg_SX,G_list_contract2)
            obs["avg_sx2"] = trace_complex(avg_SX)/norm

        
        # prepare list with labels and values
        obs_labels=["avg_SS1"]+["avg_SS2"]+["avg_sx1"]+["avg_sx2"]+["avg_sz1"]+["avg_sz2"]+["avg_SSx1"]+["avg_SSx2"]+["avg_SSz1"]+["avg_SSz2"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,coord,direction,state,env,dist):
   
        # function allowing for additional site-dependent conjugation of op
        def conjugate_op(op):
            #rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
            rot_op= torch.eye(self.phys_dim, dtype=self.dtype, device=self.device)
            op_0= op
            op_rot= torch.einsum('ki,kl,lj->ij',rot_op,op_0,rot_op)
            def _gen_op(r):
                #return op_rot if r%2==0 else op_0
                return op_0
            return _gen_op

        op_sx= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
        op_isy= -0.5*(self.obs_ops["sp"] - self.obs_ops["sm"]) 

        Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, self.obs_ops["sz"], \
            conjugate_op(self.obs_ops["sz"]), dist)
        Sx0sxR= corrf.corrf_1sO1sO(coord,direction,state,env, op_sx, conjugate_op(op_sx), dist)
        nSy0SyR= corrf.corrf_1sO1sO(coord,direction,state,env, op_isy, conjugate_op(op_isy), dist)

        res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res  

class J1J2_C4V_BIPARTITE():
    def __init__(self, j1=1.0, j2=0.0, global_args=cfg.global_args):
        r"""
        :param j1: nearest-neighbour interaction
        :param j2: next nearest-neighbour interaction
        :param global_args: global configuration
        :type j1: float
        :type j2: float
        :type global_args: GLOBALARGS

        Build Spin-1/2 :math:`J_1-J_2` Hamiltonian

        .. math:: 

            H = J_1\sum_{<i,j>} \mathbf{S}_i.\mathbf{S}_j + J_2\sum_{<<i,j>>} \mathbf{S}_i.\mathbf{S}_j
            = \sum_{p} h_p

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), and the second sum runs over 
        pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`)::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
                :  :  :  :

        where

        * :math:`h_p = J_1(\mathbf{S}_{r}.\mathbf{S}_{r+\vec{x}} + \mathbf{S}_{r}.\mathbf{S}_{r+\vec{y}})
          +J_2(\mathbf{S}_{r}.\mathbf{S}_{r+\vec{x}+\vec{y}} + \mathbf{S}_{r+\vec{x}}.\mathbf{S}_{r+\vec{y}})` 
          with indices of spins ordered as follows :math:`s_r s_{r+\vec{x}} s_{r+\vec{y}} s_{r+\vec{x}+\vec{y}};
          s'_r s'_{r+\vec{x}} s'_{r+\vec{y}} s'_{r+\vec{x}+\vec{y}}`

        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.j1=j1
        self.j2=j2
        
        self.h2, self.h2_rot, self.hp = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        id2= torch.eye(self.phys_dim**2,dtype=self.dtype,device=self.device)
        id2= id2.view(tuple([self.phys_dim]*4)).contiguous()
        expr_kron = 'ij,ab->iajb'
        SS= torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
            + torch.einsum(expr_kron,s2.SM(),s2.SP()))
        rot_op= s2.BP_rot()
        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,SS,rot_op)

        h2x2_SS= torch.einsum('ijab,klcd->ijklabcd',SS,id2) # nearest neighbours
        # 0 1     0 1   0 x   x x   x 1
        # 2 3 ... x x + 2 x + 2 3 + x 3
        hp= 0.5*self.j1*(h2x2_SS + h2x2_SS.permute(0,2,1,3,4,6,5,7)\
           + h2x2_SS.permute(2,3,0,1,6,7,4,5) + h2x2_SS.permute(3,1,2,0,7,5,6,4)) \
           + self.j2*(h2x2_SS.permute(0,3,2,1,4,7,6,5) + h2x2_SS.permute(2,1,0,3,6,5,4,7))
        hp= torch.einsum('xj,yk,ixylauvd,ub,vc->ijklabcd',rot_op,rot_op,hp,rot_op,rot_op)
        hp= hp.contiguous()
        return SS, SS_rot, hp

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= s2.SZ()
        obs_ops["sp"]= s2.SP()
        obs_ops["sm"]= s2.SM()
        return obs_ops

    def energy_1x1(self,state,env_c4v,**kwargs):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS
        :type env_c4v: ENV_C4V
        :return: energy per site
        :rtype: float

        We assume 1x1 C4v iPEPS which tiles the lattice with a bipartite pattern composed 
        of two tensors A, and B=RA, where R rotates approriately the physical Hilbert space 
        of tensor A on every "odd" site::

            1x1 C4v => rotation P => BIPARTITE

            A A A A                  A B A B
            A A A A                  B A B A
            A A A A                  A B A B
            A A A A                  B A B A

        Due to C4v symmetry it is enough to construct a single reduced density matrix 
        :py:func:`ctm.one_site_c4v.rdm_c4v.rdm2x2` of a 2x2 plaquette. Afterwards, 
        the energy per site `e` is computed by evaluating a single plaquette term :math:`h_p`
        containing two nearest-nighbour terms :math:`\bf{S}.\bf{S}` and two next-nearest 
        neighbour :math:`\bf{S}.\bf{S}`, as:

        .. math::

            e = \langle \mathcal{h_p} \rangle = Tr(\rho_{2x2} \mathcal{h_p})
        
        """
        rdm2x2= rdm_c4v.rdm2x2(state,env_c4v,sym_pos_def=True,\
            verbosity=cfg.ctm_args.verbosity_rdm)
        energy_per_site= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.hp)
        return energy_per_site

    def energy_1x1_lowmem(self, state, env_c4v, force_cpu=False):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS
        :type env_c4v: ENV_C4V
        :return: energy per site
        :rtype: float

        We assume 1x1 C4v iPEPS which tiles the lattice with a bipartite pattern composed 
        of two tensors A, and B=RA, where R rotates approriately the physical Hilbert space 
        of tensor A on every "odd" site::

            1x1 C4v => rotation P => BIPARTITE

            A A A A                  A B A B
            A A A A                  B A B A
            A A A A                  A B A B
            A A A A                  B A B A

        Due to C4v symmetry it is enough to construct two reduced density matrices.
        In particular, :py:func:`ctm.one_site_c4v.rdm_c4v.rdm2x1` of a NN-neighbour pair
        and :py:func:`ctm.one_site_c4v.rdm_c4v.rdm2x1_diag` of NNN-neighbour pair. 
        Afterwards, the energy per site `e` is computed by evaluating a term :math:`h2_rot`
        containing :math:`\bf{S}.\bf{S}` for nearest- and :math:`h2` term for 
        next-nearest- expectation value as:

        .. math::

            e = 2*\langle \mathcal{h2} \rangle_{NN} + 2*\langle \mathcal{h2} \rangle_{NNN}
            = 2*Tr(\rho_{2x1} \mathcal{h2_rot}) + 2*Tr(\rho_{2x1_diag} \mathcal{h2})
        
        """
        rdm2x2_NN= rdm_c4v.rdm2x2_NN_lowmem_sl(state, env_c4v, sym_pos_def=True,\
            force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
        rdm2x2_NNN= rdm_c4v.rdm2x2_NNN_lowmem_sl(state, env_c4v, sym_pos_def=True,\
            force_cpu=force_cpu, verbosity=cfg.ctm_args.verbosity_rdm)
        energy_per_site= 2.0*self.j1*torch.einsum('ijkl,ijkl',rdm2x2_NN,self.h2_rot)\
            + 2.0*self.j2*torch.einsum('ijkl,ijkl',rdm2x2_NNN,self.h2)
        return energy_per_site

    def eval_obs(self,state,env_c4v,force_cpu=False):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS
        :type env_c4v: ENV_C4V
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. magnetization
            2. :math:`\langle S^z \rangle,\ \langle S^+ \rangle,\ \langle S^- \rangle`
    
        where the on-site magnetization is defined as
        
        .. math::
            
            \begin{align*}
            m &= \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }
            =\sqrt{\langle S^z \rangle^2+1/4(\langle S^+ \rangle+\langle S^- 
            \rangle)^2 -1/4(\langle S^+\rangle-\langle S^-\rangle)^2} \\
              &=\sqrt{\langle S^z \rangle^2 + 1/2\langle S^+ \rangle \langle S^- \rangle)}
            \end{align*}

        Usual spin components can be obtained through the following relations
        
        .. math::
            
            \begin{align*}
            S^+ &=S^x+iS^y               & S^x &= 1/2(S^+ + S^-)\\
            S^- &=S^x-iS^y\ \Rightarrow\ & S^y &=-i/2(S^+ - S^-)
            \end{align*}
        """
        # TODO optimize/unify ?
        # expect "list" of (observable label, value) pairs ?
        obs= dict()
        with torch.no_grad():
            rdm2x1= rdm_c4v.rdm2x1_sl(state,env_c4v,force_cpu=force_cpu,\
                verbosity=cfg.ctm_args.verbosity_rdm)
            obs[f"SS2x1"]= torch.einsum('ijab,ijab',rdm2x1,self.h2_rot)
            
            # reduce rdm2x1 to 1x1
            rdm1x1= torch.einsum('ijaj->ia',rdm2x1)
            rdm1x1= rdm1x1/torch.trace(rdm1x1)
            for label,op in self.obs_ops.items():
                obs[f"{label}"]= torch.trace(rdm1x1@op)
            obs[f"m"]= sqrt(abs(obs[f"sz"]**2 + obs[f"sp"]*obs[f"sm"]))
            
        # prepare list with labels and values
        obs_labels=[f"m"]+[f"{lc}" for lc in self.obs_ops.keys()]+[f"SS2x1"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,state,env_c4v,dist,canonical=False):
        Sop_zxy= torch.zeros((3,self.phys_dim,self.phys_dim),dtype=self.dtype,device=self.device)
        Sop_zxy[0,:,:]= self.obs_ops["sz"]
        Sop_zxy[1,:,:]= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
        Sop_zxy[2,:,:]= -0.5*(self.obs_ops["sp"] - self.obs_ops["sm"])

        # compute vector of spontaneous magnetization
        if canonical:
            s_vec_zpm=[]
            rdm1x1= rdm_c4v.rdm1x1(state,env_c4v)
            for label in ["sz","sp","sm"]:
                op= self.obs_ops[label]
                s_vec_zpm.append(torch.trace(rdm1x1@op))
            # 0) transform into zxy basis and normalize
            s_vec_zxy= torch.tensor([s_vec_zpm[0],0.5*(s_vec_zpm[1]+s_vec_zpm[2]),\
                0.5*(s_vec_zpm[1]-s_vec_zpm[2])],dtype=self.dtype,device=self.device)
            s_vec_zxy= s_vec_zxy/torch.norm(s_vec_zxy)
            # 1) build rotation matrix
            R= torch.tensor([[s_vec_zxy[0],-s_vec_zxy[1],0],[s_vec_zxy[1],s_vec_zxy[0],0],[0,0,1]],\
                dtype=self.dtype,device=self.device).t()
            # 2) rotate the vector of operators
            Sop_zxy= torch.einsum('ab,bij->aij',R,Sop_zxy)

        # function generating properly rotated operators on every bi-partite site
        def get_bilat_op(op):
            rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
            op_0= op
            op_rot= torch.einsum('ki,kl,lj->ij',rot_op,op_0,rot_op)
            def _gen_op(r):
                return op_rot if r%2==0 else op_0
            return _gen_op

        Sz0szR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[0,:,:], \
            get_bilat_op(Sop_zxy[0,:,:]), dist)
        Sx0sxR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[1,:,:], get_bilat_op(Sop_zxy[1,:,:]), dist)
        nSy0SyR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[2,:,:], get_bilat_op(Sop_zxy[2,:,:]), dist)

        res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res

    def eval_corrf_DD_H(self,state,env_c4v,dist,verbosity=0):
        # function generating properly rotated S.S operator on every bi-partite site
        rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
        # (S.S)_s1s2,s1's2' with rotation applied on "first" spin s1,s1' 
        SS_rot= torch.einsum('ki,kjcb,ca->ijab',rot_op,self.h2,rot_op)
        # (S.S)_s1s2,s1's2' with rotation applied on "second" spin s2,s2'
        op_rot= SS_rot.permute(1,0,3,2).contiguous()
        def _gen_op(r):
            return SS_rot if r%2==0 else op_rot
        
        D0DR= corrf_c4v.corrf_2sOH2sOH_E1(state, env_c4v, SS_rot, _gen_op, dist, verbosity=verbosity)

        res= dict({"dd": D0DR})
        return res
