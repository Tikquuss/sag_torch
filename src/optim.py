# 
import re
import inspect
from torch import optim

from .optimizers.sag import SimpleSAG, SAGBase, SAGWithd
from .optimizers.sgd import CustomSGD, SAGSGDBase, SAGSGDWithd
from .optimizers.adam import CustomAdam, \
    SAGAdamSimpleBase, SAGAdamBase, \
    SAGAdamSimpleWithd, SAGAdamWithd, \
    AdamCosineWithWarmup, AdamInverseSqrtWithWarmup

def get_all_optims(weight_decay=0.0, momentum= 0.9, beta1=0.9, beta2=0.99):
    all_optims = {
    "sgd" : f"sgd,momentum=0,dampening=0,weight_decay={weight_decay},nesterov=False",
    "momentum" : f"sgd,momentum={momentum},dampening=0.9,weight_decay={weight_decay},nesterov=False",
    "nesterov" : f"sgd,momentum={momentum},dampening=0,weight_decay={weight_decay},nesterov=True",
    "asgd" : f"asgd,lambd=0.0001,alpha=0.75,t0=1000000.0,weight_decay={weight_decay}",
    "rmsprop" : f"rmsprop,alpha=0.99,weight_decay={weight_decay},momentum=0,centered=False",
    "rmsprop_mom" : f"rmsprop,alpha=0.99,weight_decay={weight_decay},momentum={momentum},centered=False",
    "rprop" : f"rprop,etaplus=0.5,etaminus=1.2,step_min=1e-06,step_max=50",
    "adadelta" : f"adadelta,rho=0.9,weight_decay={weight_decay}", 
    "adagrad" : f"adagrad,lr_decay=0,weight_decay={weight_decay},initial_accumulator_value=0", 
    "adam" : f"adam,weight_decay={weight_decay},beta1={beta1},beta2={beta2},amsgrad=False",
    "amsgrad" : f"adam,weight_decay={weight_decay},beta1={beta1},beta2={beta2},amsgrad=True",
    "adamax" : f"adamax,weight_decay={weight_decay},beta1={beta1},beta2={beta2}",
    "custom_adam" : f"custom_adam,weight_decay={weight_decay},beta1={beta1},beta2={beta2}",
    "adam_inverse_sqrt" : f"adam_inverse_sqrt,weight_decay={weight_decay},beta1={beta1},beta2={beta2},warmup_updates=4000,warmup_init_lr=1e-7,exp_factor=0.5",
    "adam_cosine" : f"adam_cosine,weight_decay={weight_decay},beta1={beta1},beta2={beta2},warmup_updates=4000,warmup_init_lr=1e-7,min_lr=1e-9,init_period=1000000,period_mult=1,lr_shrink=0.75"
    }
    return all_optims

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_params_from_string(s : str, separator = ",", have_method=True):
    re_float="^[+-]?(\d+(\.\d*)?|\.\d+)$"
    # https://stackoverflow.com/a/41668652/11814682
    re_scient="[+\-]?[^A-Za-z]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)"
    if "," in s:
        if have_method :
            method = s[:s.find(separator)]
            s = s[s.find(separator) + 1:]
        else :
            method = ""
        optim_params = {}
        for x in s.split(separator):
            split = x.split('=')
            assert len(split) == 2
            try:
                float(split[1])
                assert (re.match(re_float, split[1]) is not None) or (re.match(re_scient, split[1]) is not None)
                optim_params[split[0]] = float(split[1])
            except ValueError:
                optim_params[split[0]] = split[1]
    else:
        method = s
        optim_params = {}
        
    return method, optim_params

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {'off', 'false', '0'}
    TRUTHY_STRINGS = {'on', 'true', '1'}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise TypeError("Invalid value for a boolean flag!")

def get_optimizer(parameters, s, noamopt=""):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.001"
        - "adagrad,lr=0.001,lr_decay=0.05"
    noamopt : Optim wrapper that implements rate.
        -"factor_ae=1,warmup_ae=200"
    """
    method, optim_params = get_params_from_string(s, separator = ",", have_method=True)
    #al = "sgd","asgd","rmsprop","rprop","adadelta","adagrad","adam","adamax","custom_adam","adam_inverse_sqrt","adam_cosine","sag"
    #al = "sgd","asgd","rmsprop","rprop","adadelta","adagrad","adam","adamax","sag"
    if method == 'sgd':
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        #"lr=,momentum=0,dampening=0,weight_decay=0,nesterov=False"
        optim_params["nesterov"] = bool_flag(optim_params.get("nesterov", 'False'))
        optim_fn = optim.SGD
        #optim_fn = CustomSGD
        assert 'lr' in optim_params
    elif method == 'asgd':
        # https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html
        # https://sci-hub.se/10.1137/0330046
        # https://www.quora.com/How-does-Averaged-Stochastic-Gradient-Decent-ASGD-work
        # lr=,lambd=0.0001,alpha=0.75,t0=1000000.0,weight_decay=0
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
        #lr=,alpha=0.99,weight_decay=0,momentum=0,centered=False
        optim_params["centered"] = bool_flag(optim_params.get("centered", 'False'))
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        # https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html
        # lr=,etas=(0.5, 1.2), step_sizes=(1e-06, 50)
        # lr=,etaplus=0.5,etaminus=1.2,step_min=1e-06,step_max=50
        optim_params['etas'] = (optim_params.pop('etaplus', 0.5), optim_params.pop('etaminus', 1.2))
        optim_params['step_sizes'] = (optim_params.pop('step_min', 1e-06), optim_params.pop('step_max', 50))
        optim_fn = optim.Rprop
    elif method == 'adadelta':
        # https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html
        # lr=,rho=0.9,weight_decay=0
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        # https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
        # lr=,lr_decay=0,weight_decay=0,initial_accumulator_value=0
        optim_fn = optim.Adagrad
    elif method in ['adam', 'adamax', 'custom_adam', 'adam_inverse_sqrt', 'adam_cosine']:
        # lr, betas,weight decay,amsgrad
        optim_params['betas'] = (optim_params.pop('beta1', 0.9), optim_params.pop('beta2', 0.999))
        if method == 'adam' :
            # lr=0.001, betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False 
            optim_params["amsgrad"] = bool_flag(optim_params.get("amsgrad", 'False'))
            optim_fn = optim.Adam
        elif method == 'adamax':
            # https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html
            # lr=0.002, betas=(0.9,0.999),eps=1e-08,weight_decay=0
            optim_fn = optim.Adamax
        elif method == 'custom_adam' :
            # lr=,betas=(0.9,0.999),eps=1e-8,weight_decay=0
            optim_fn = CustomAdam 
        elif method == 'adam_inverse_sqrt':
            # lr=,betas=(0.9,0.999),eps=1e-8,weight_decay=0,warmup_updates=4000,warmup_init_lr=1e-7,exp_factor=0.5
            optim_fn = AdamInverseSqrtWithWarmup
        elif method == 'adam_cosine':
            # lr=,betas=(0.9, 0.999),eps=1e-8,weight_decay=0,warmup_updates=4000,warmup_init_lr=1e-7,min_lr=1e-9,init_period=1000000,period_mult=1,lr_shrink=0.75
            optim_fn = AdamCosineWithWarmup
    elif "sag" in method :
        if method == 'sagbase':
            optim_fn = SAGBase
        else :
            assert 'batch_mode' in optim_params
            assert 'init_y_i' in optim_params
            optim_params["batch_mode"] = bool_flag(optim_params["batch_mode"])
            optim_params["init_y_i"] = bool_flag(optim_params["init_y_i"])
            optim_params["sum_all"] = bool_flag(optim_params.get("sum_all", 'False'))
            with_d = bool_flag(optim_params.pop("with_d", 'True'))
            if method == 'sag' : 
                # good without decay
                # SAGWithd is slightly better than SAGBase
                optim_fn = SAGWithd if with_d else SAGBase 
            elif method == 'sag_sgd' :
                optim_params["nesterov"] = bool_flag(optim_params.get("nesterov", 'False'))
                # good without decay
                # SAGSGDWithd is slightly better than SAGSGDBase
                optim_fn =  SAGSGDWithd if with_d else SAGSGDBase
            elif method == 'sag_adam' :
                optim_params['betas'] = (optim_params.pop('beta1', 0.9), optim_params.pop('beta2', 0.999))
                #optim_params['weight_decay'] = 1.0 
                if with_d :
                    optim_fn = SAGAdamSimpleWithd # good, but unstable
                    #optim_fn = SAGAdamWithd # good, but unstable
                else :
                    optim_fn = SAGAdamSimpleBase # good
                    #optim_fn = SAGAdamBase # bad
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    try : 
        expected_args = inspect.getargspec(optim_fn.__init__)[0]
    except ValueError: #Function has keyword-only parameters or annotations, use getfullargspec() API which can support them
        expected_args = inspect.getfullargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    if noamopt != "" :
        _, params = get_params_from_string(s=noamopt, have_method=False)
        return NoamOpt(params["d_model"], params["factor_ae"], params["warmup_ae"], optim_fn(parameters, **optim_params))
    else :
        return optim_fn(parameters, **optim_params)


def get_lr_scheduler(optimizer, s) :
    """
    Parse lr_scheduler parameters.
    Input should be of the form:
        - "reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss"

    """
    method, scheduler_params = get_params_from_string(s, separator = ",", have_method=True)
    monitor = scheduler_params.pop("monitor", "val_loss")
    if method == "reduce_lr_on_plateau" :
        # Reduce learning rate when a metric has stopped improving.
        scheduler_fn =  optim.lr_scheduler.ReduceLROnPlateau
    elif method == "constant_lr" :
        # Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches a pre-defined milestone: total_iters.
        scheduler_fn =  optim.lr_scheduler.ConstantLR
    elif method == "linear_lr" :
        # Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
        scheduler_fn =  optim.lr_scheduler.LinearLR
    elif method == "cosine_annealing_lr" :
        # Set the learning rate of each parameter group using a cosine annealing schedule, where η_max is set to the initial lr and T_{cur} is the number of epochs since the last restart in SGDR
        scheduler_fn =  optim.lr_scheduler.CosineAnnealingLR
    elif method == "exponential_lr" :
        # Decays the learning rate of each parameter group by gamma every epoch.
        scheduler_fn =  optim.lr_scheduler.ExponentialLR
    elif method == "lambda_lr" :
        # Sets the learning rate of each parameter group to the initial lr times a given function.
        scheduler_fn = optim.lr_scheduler.LambdaLR
    elif method == "multiplicative_lr" :
        # Multiply the learning rate of each parameter group by the factor given in the specified function.
        scheduler_fn = optim.lr_scheduler.MultiplicativeLR
    elif method == "step_lr" :
        # Decays the learning rate of each parameter group by gamma every step_size epochs.
        scheduler_fn = optim.lr_scheduler.StepLR
    elif method == "multi_step_lr" :
        # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
        scheduler_fn = optim.lr_scheduler.MultiStepLR
    elif method == "cyclic_lr" :
        # Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).
        scheduler_fn = optim.lr_scheduler.CyclicLR
    elif method == "one_cycle_lr" :
        # Sets the learning rate of each parameter group according to the 1cycle learning rate policy.
        scheduler_fn = optim.lr_scheduler.OneCycleLR
    elif method == "cosine_annealing_warm_restarts" :
        # Set the learning rate of each parameter group using a cosine annealing schedule, where η_max is set to the initial lr, T_cur is the number of epochs since the last restart and 
        # T_i is the number of epochs between two warm restarts in SGDR
        scheduler_fn = optim.lr_scheduler.CosineAnnealingWarmRestarts
    # elif method == "chained_scheduler" :
    #     # Chains list of learning rate schedulers.
    #     scheduler_fn = optim.lr_scheduler.ChainedScheduler
    # elif method == "sequential_lr" :
    #     # Receives the list of schedulers that is expected to be called sequentially during optimization process and milestone points that provides exact intervals to reflect which scheduler is supposed to be called at a given epoch.
    #     scheduler_fn = optim.lr_scheduler.SequentialLR
    else:
        raise Exception('Unknown lr scheduler method: "%s"' % method)

    # check that we give good parameters to the optimizer
    try : 
        expected_args = inspect.getargspec(scheduler_fn.__init__)[0]
    except ValueError: #Function has keyword-only parameters or annotations, use getfullargspec() API which can support them
        expected_args = inspect.getfullargspec(scheduler_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'optimizer']
    if not all(k in expected_args[2:] for k in scheduler_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(scheduler_params.keys())))

    return scheduler_fn(optimizer, **scheduler_params), monitor

def configure_optimizers(parameters, s_optim, s_lr_scheduler):
    optimizer = get_optimizer(parameters, s_optim)
    if s_lr_scheduler :
        scheduler, monitor = get_lr_scheduler(optimizer, s_lr_scheduler)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    else :
        return {"optimizer": optimizer}


