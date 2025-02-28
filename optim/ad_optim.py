import time
import json
import logging
log = logging.getLogger(__name__)
import torch
#from memory_profiler import profile
import config as cfg

def store_checkpoint(checkpoint_file, state, optimizer, current_epoch, current_loss):
    r"""
    :param checkpoint_file: target file
    :param state: ipeps wavefunction
    :param optimizer: Optimizer
    :param current_epoch: current epoch
    :param current_loss: current value of a loss function
    :type checkpoint_file: str or Path
    :type state: IPEPS
    :type optimizer: torch.optim.Optimizer
    :type current_epoch: int
    :type current_loss: float

    Store the current state of the optimization in ``checkpoint_file``.
    """
    torch.save({
            'epoch': current_epoch,
            'loss': current_loss,
            'parameters': state.get_checkpoint(),
            'optimizer_state_dict': optimizer.state_dict()}, checkpoint_file)

def optimize_state(state, loss_fn, obs_fn=None, post_proc=None,
    main_args=cfg.main_args, opt_args=cfg.opt_args, ctm_args=cfg.ctm_args, 
    global_args=cfg.global_args):
    r"""
    :param state: initial wavefunction
    :param ctm_env_init: initial environment of ``state``
    :param loss_fn: loss function
    :param obs_fn: optional function to evaluate observables
    :param post_proc: optional function for post-processing the state and environment  
    :param main_args: main configuration
    :param opt_args: optimization configuration
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS
    :type ctm_env_init: ENV
    :type loss_fn: function(IPEPS,ENV,dict)->torch.tensor
    :type obs_fn: function(IPEPS,ENV,dict)->None
    :type post_proc: function(IPEPS,ENV,dict)->None
    :type main_args: MAINARGS
    :type opt_args: OPTARGS
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Optimizes initial wavefunction ``state`` with respect to ``loss_fn`` using 
    `LBFGS optimizer <https://pytorch.org/docs/stable/optim.html#torch.optim.LBFGS>`_.
    The main parameters influencing the optimization process are given in :class:`config.OPTARGS`.
    Calls to functions ``loss_fn``, ``obs_fn``, and ``post_proc`` pass the current configuration
    as dictionary ``{"ctm_args":ctm_args, "opt_args":opt_args}``.
    """
    verbosity = opt_args.verbosity_opt_epoch
    checkpoint_file = main_args.out_prefix+"_checkpoint.p"   
    outputstatefile= main_args.out_prefix+"_state.json"
    t_data = dict({"loss": [], "min_loss": float('inf')})
    context= dict({"opt_args":opt_args, "loss_history": t_data})
    epoch= 0

    parameters= state.get_parameters()
    for A in parameters: A.requires_grad_(True)

    optimizer = torch.optim.LBFGS(parameters, max_iter=opt_args.max_iter_per_epoch, lr=opt_args.lr, \
        tolerance_grad=opt_args.tolerance_grad, tolerance_change=opt_args.tolerance_change, \
        history_size=opt_args.history_size)

    # load and/or modify optimizer state from checkpoint
    if main_args.opt_resume is not None:
        print(f"INFO: resuming from check point. resume = {main_args.opt_resume}")
        checkpoint = torch.load(main_args.opt_resume)
        epoch0 = checkpoint["epoch"]
        loss0 = checkpoint["loss"]
        cp_state_dict= checkpoint["optimizer_state_dict"]
        cp_opt_params= cp_state_dict["param_groups"][0]
        cp_opt_history= cp_state_dict["state"][cp_opt_params["params"][0]]
        if main_args.opt_resume_override_params:
            cp_opt_params["lr"] = opt_args.lr
            cp_opt_params["max_iter"] = opt_args.max_iter_per_epoch
            cp_opt_params["tolerance_grad"] = opt_args.tolerance_grad
            cp_opt_params["tolerance_change"] = opt_args.tolerance_change
            # resize stored old_dirs, old_stps, ro, al to new history size
            cp_history_size= cp_opt_params["history_size"]
            cp_opt_params["history_size"] = opt_args.history_size
            if opt_args.history_size < cp_history_size:
                if len(cp_opt_history["old_dirs"]) > opt_args.history_size: 
                    cp_opt_history["old_dirs"]= cp_opt_history["old_dirs"][-opt_args.history_size:]
                    cp_opt_history["old_stps"]= cp_opt_history["old_stps"][-opt_args.history_size:]
            cp_ro_filtered= list(filter(None,cp_opt_history["ro"]))
            cp_al_filtered= list(filter(None,cp_opt_history["al"]))
            if len(cp_ro_filtered) > opt_args.history_size:
                cp_opt_history["ro"]= cp_ro_filtered[-opt_args.history_size:]
                cp_opt_history["al"]= cp_al_filtered[-opt_args.history_size:]
            else:
                cp_opt_history["ro"]= cp_ro_filtered + [None for i in range(opt_args.history_size-len(cp_ro_filtered))]
                cp_opt_history["al"]= cp_al_filtered + [None for i in range(opt_args.history_size-len(cp_ro_filtered))]
        cp_state_dict["param_groups"][0]= cp_opt_params
        cp_state_dict["state"][cp_opt_params["params"][0]]= cp_opt_history
        optimizer.load_state_dict(cp_state_dict)
        print(f"checkpoint.loss = {loss0}")
    
    prev_loss = None

    #@profile
    def closure():
        # 0) evaluate loss
        nonlocal prev_loss
        optimizer.zero_grad()
        loss = loss_fn(state, context)
        loss_value = loss.item()
        if torch.isnan(loss):
            print("WARNING: NaN detected in loss. Stopping optimization.")
            t_data["loss"].append(float('nan'))
            return loss  # NaNが発生したら loss を返して step を止める
        # エネルギー差を計算
        energy_diff = abs(prev_loss - loss_value) if prev_loss is not None else None
        prev_loss = loss_value  # 今回のエネルギーを記録
        # 1) store current state if the loss improves
        t_data["loss"].append(loss.item())
        if t_data["min_loss"] > loss_value:
            # Loss improved -> Reset stable_count
            t_data["min_loss"] = loss_value
            t_data["stable_count"] = 0
            state.write_to_file(outputstatefile, normalize=True)
        elif len(t_data["loss"]) > 1 and t_data["loss"][-2] == loss_value:
            # Loss unchanged -> Increment stable_count
            t_data["stable_count"] += 1
        else:
            # Loss increased -> Reset stable_count
            t_data["stable_count"] = 0
        

        # 3) compute desired observables
        if obs_fn is not None:
            obs_fn(state, context)

        # 4) evaluate gradient
        t_grad0= time.perf_counter()
        loss.backward()
        t_grad1= time.perf_counter()

        # 5) log grad metrics for debugging
        if opt_args.opt_logging:
            log_entry=dict({"id": epoch, "t_grad": t_grad1-t_grad0, "energy_diff": energy_diff})
            if opt_args.opt_log_grad: log_entry["grad"]= [p.grad.tolist() for p in parameters]
            log.info(json.dumps(log_entry))
        
        return loss
    
    for epoch in range(main_args.opt_max_iter):
        # checkpoint the optimizer
        # checkpointing before step, guarantees the correspondence between the wavefunction
        # and the last computed value of loss t_data["loss"][-1]
        if epoch>0:
            store_checkpoint(checkpoint_file, state, optimizer, epoch, t_data["loss"][-1])

        # After execution closure ``current_env`` **IS NOT** corresponding to ``state``, since
        # the ``state`` on-site tensors have been modified by gradient. 
        optimizer.step(closure)
        
        if post_proc is not None:
            post_proc(state, context)
            
        if t_data["stable_count"] >= 20:
            print(f"INFO: Loss has remained unchanged for 20 consecutive iterations. Stopping optimization.")
            break

        if len(t_data["loss"]) > 1 and torch.isnan(torch.tensor(t_data["loss"][-1])):
            print("ERROR: NaN detected in loss history. Optimization terminated.")
            break

    # optimization is over, store the last checkpoint
    store_checkpoint(checkpoint_file, state, optimizer, \
        main_args.opt_max_iter, t_data["loss"][-1])
