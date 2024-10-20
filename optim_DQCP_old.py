#python optim_DQCP_old.py --tiling 1SITE --bond_dim 10 --seed 123 --Jz 1.4 --N 30 --CTMARGS_fwd_checkpoint_move --OPTARGS_tolerance_grad 1.0e-8 --out_prefix ex-mps_s30_chi10_jz14
import context
import scipy.io
import torch
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

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--Jx", type=float, default=1., help="nearest-neighbour Sx coupling")
parser.add_argument("--Jz", type=float, default=0., help="nearest-neighbour Sz coupling")
parser.add_argument("--Kx", type=float, default=0.5, help="next-nearest-neighbour Sx coupling")
parser.add_argument("--Kz", type=float, default=0.5, help="next-nearest-neighbour Sz coupling")
parser.add_argument("--N", type=int, default=10, help="number of spins")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice")
args, unknown_args = parser.parse_known_args()

#mat = scipy.io.loadmat('tensor_D_24.mat')
#print (mat['Al'].transpose(1,0,2).shape)

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model = DQCP.DQCP(Jx=args.Jx, Jz=args.Jz, Kx=args.Kx, Kz=args.Kz, N=args.N)
    
    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz    
    if args.tiling == "1SITE":
        def lattice_to_site(coord):
            return (0)
    elif args.tiling == "2SITE":
        def lattice_to_site(coord):
            return (coord[0]%2)
            
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"1SITE"+"2SITE")
    #bond_dim = args.bond_dim
    #A1 = torch.as_tensor(mat['Al'].transpose(1,0,2),\
    #    dtype=cfg.global_args.dtype,device=cfg.global_args.device)
    #A2 = torch.zeros((model.phys_dim, bond_dim, bond_dim),\
    #    dtype=cfg.global_args.dtype,device=cfg.global_args.device)
    #A = torch.zeros((2, model.phys_dim, bond_dim, bond_dim),\
    #    dtype=cfg.global_args.dtype,device=cfg.global_args.device)
    #A[0] = A1; A[1] = A2
    #A = A/max_complex(A)
    #sites = {(0): A}
            
    #state = MPS(sites, vertexToSite=lattice_to_site)
    if args.instate!=None:
        state = read_ipeps(args.instate, vertexToSite=lattice_to_site)
        print (state.sites[(0)])
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim

        if args.tiling == "1SITE":
            A1 = torch.rand((model.phys_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            A2 = torch.rand((model.phys_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            A = torch.zeros((2, model.phys_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            A[0] = A1#; A[1] = A2
            A = A/max_complex(A)
            
            sites = {(0): A}
        elif args.tiling == "2SITE":
            A1 = torch.rand((model.phys_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            A2 = torch.rand((model.phys_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            A = torch.zeros((2, model.phys_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            A[0] = A1#; A[1] = A2
            A = A/max_complex(A)
            B1 = torch.rand((model.phys_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            B2 = torch.rand((model.phys_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            B = torch.zeros((2, model.phys_dim, bond_dim, bond_dim),\
                dtype=cfg.global_args.dtype,device=cfg.global_args.device)
            B[0] = B1#; B[1] = B2
            B = B/max_complex(B)
            
            sites = {(0): A, (1): B}
            
        state = MPS(sites, vertexToSite=lattice_to_site)
        
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    # 2) select the "energy" function 
    if args.tiling == "1SITE":
        energy_f=model.energy_2x2_1site
    elif args.tiling == "2SITE":
        energy_f=model.energy_2x2_2site
    else:
        raise ValueError("Invalid tiling: "+str(args.tiling)+" Supported options: "\
            +"BIPARTITE, 2SITE, 4SITE, 8SITE")

    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = energy_f(state)
        history.append(e_curr.item())

        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    
    loss0 = energy_f(state)
    obs_values, obs_labels = model.eval_obs(state)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, opt_context):
        opt_args= opt_context["opt_args"]

        # 1) evaluate loss with the converged environment
        loss = energy_f(state)
        
        return (loss)

    @torch.no_grad()
    def obs_fn(state, opt_context):
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = model.eval_obs(state)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))
            log.info("Norm(sites): "+", ".join([f"{t.norm()}" for c,t in state.sites.items()]))

    # optimize
    optimize_state(state, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps(outputstatefile, vertexToSite=state.vertexToSite)
    opt_energy = energy_f(state)
    obs_values, obs_labels = model.eval_obs(state)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))  

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j2=0.0
        args.bond_dim=2
        args.chi=16
        args.opt_max_iter=3
        try:
            import scipy.sparse.linalg
            self.SCIPY= True
        except:
            print("Warning: Missing scipy. Arnoldi methods not available.")
            self.SCIPY= False

    # basic tests
    def test_opt_GESDD_BIPARTITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    def test_opt_GESDD_BIPARTITE_LS_strong_wolfe(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        args.line_search="strong_wolfe"
        main()

    def test_opt_GESDD_BIPARTITE_LS_backtracking(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        args.line_search="backtracking"
        args.line_search_svd_method="ARP"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_BIPARTITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        main()

    def test_opt_GESDD_BIPARTITE_LS_backtracking_gpu(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="BIPARTITE"
        args.line_search="backtracking"
        args.line_search_svd_method="ARP"
        main()

    def test_opt_GESDD_4SITE(self):
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_GESDD_4SITE_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="GESDD"
        args.tiling="4SITE"
        main()
