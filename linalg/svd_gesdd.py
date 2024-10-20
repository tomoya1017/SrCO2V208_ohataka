'''
Implementation taken from https://arxiv.org/abs/1903.09650
which follows derivation given in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
'''
import torch
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import config as cfg
from complex_num.complex_operation import *
from scipy.sparse.linalg import LinearOperator

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class SVDGESDD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        AA = A.detach().cpu().numpy()
        A2 = AA[0] + 1j*AA[1]
        UU, SS, VV = linalg.svd(A2)#, k=A2.shape[0]-2)#, lapack_driver="gesdd")
        
        U1 = torch.as_tensor(UU.real)#.clone()
        U2 = torch.as_tensor(UU.imag)#.clone()
        U = torch.stack((U1, U2), dim=0)
        
        V1 = torch.as_tensor(VV.real.transpose())#.clone()
        V2 = torch.as_tensor(-VV.imag.transpose())#.clone()
        V = torch.stack((V1, V2), dim=0)
        
        S1 = torch.as_tensor(SS)#.clone()
        dim = S1.size()
        temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        S = torch.stack((S1, temp), dim=0)
        

        #S[0],p= torch.sort(S[0],descending=True)
        #U[0]= U[0][:,p]
        #U[1]= U[1][:,p]
        #V[0]= V[0][:,p]
        #V[1]= V[1][:,p]
        
        
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = complex_conjugate(transpose_complex(V))
        Ut = complex_conjugate(transpose_complex(U))
        M = size_complex(U)[0]#.size(0)
        N = size_complex(V)[0]#.size(0)
        NS = len(S[0])
        
        S2 = S[0]
        F = (S2 - S2[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S2 + S2[:, None])
        G.diagonal().fill_(float('inf'))
        G = 1/G

        Iden = torch.eye(S2.size(0), dtype=cfg.global_args.dtype, device=cfg.global_args.device)

        SIr = 1/S2
        dim = SIr.size()
        temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        SI = torch.stack((SIr, temp), dim=0)

        UdU = mm_complex(Ut, dU)#Ut @ dU
        VdV = mm_complex(Vt, dV)#Vt @ dV
        UdU2 = (UdU - complex_conjugate(transpose_complex(UdU)))/2.0
        VdV2 = (VdV - complex_conjugate(transpose_complex(VdV)))/2.0

        Jr = (F+G)*UdU2[0]; Ji = (F+G)*UdU2[1]
        J = torch.stack((Jr, Ji), dim=0)
        #JJt = J + complex_conjugate(transpose_complex(J))
        #JJtS = mm_complex(J, diag_complex(S))
        Kr = (F-G)*VdV2[0]; Ki = (F-G)*VdV2[1]
        K = torch.stack((Kr, Ki), dim=0)
        #KKt = K + complex_conjugate(transpose_complex(K))
        #SKKt = mm_complex(diag_complex(S), K)
        #print (dS)
        Lr = Iden*VdV[0]; Li = Iden*VdV[1]
        L = torch.stack((Lr, Li), dim=0)
        LLt = complex_conjugate(transpose_complex(L)) - L
        SILLt = mm_complex(diag_complex(SI), LLt)/2.0

        #dA = mm_complex(U, Vt)
        #dA = mm_complex(mm_complex(U, (J + K + diag_complex(dS))), Vt)
        dA = mm_complex(mm_complex(U, (J + K + SILLt + diag_complex(dS))), Vt)
        #print (dA[0][0][0].item())
        #U @ (Su + Sv + torch.diag(dS)) @ Vt 
        #if (M>NS):
        #    dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        #if (N>NS):
        #    dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        return dA

def test_SVDSYMEIG_random():
    M, N = 50, 40
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVDGESDD.apply, input, eps=1e-6, atol=1e-4))

if __name__=='__main__':
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    test_SVDSYMEIG_random()
