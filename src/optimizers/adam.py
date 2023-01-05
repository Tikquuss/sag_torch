
import math
import torch
from torch import optim

from .sag import SAGBase, SAGWithd

class CustomAdam(optim.Optimizer):
    """
    Same as https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py,
    without amsgrad, with step in a tensor, and states initialization in __init__.
    It was important to add `.item()` in `state['step'].item()`.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        self.check_params(lr, eps, betas)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0  # torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super().__setstate__(state)

    @staticmethod
    def check_params(lr, eps, betas):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        #if not 0.0 <= betas[0] < 1.0:
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        #if not 0.0 <= betas[1] < 1.0:
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

    def step(self, closure=None):
        """
        Step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                
                state = self.state[p]
                state['step'] += 1

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # if group['weight_decay'] != 0:
                #     grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                #exp_avg.mul_(beta1).add_(1 - beta1, grad)
                #exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # Thanks https://github.com/clovaai/AdamP/issues/5
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1) 
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) 
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                # denom = exp_avg_sq.sqrt().clamp_(min=group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']  # .item()
                bias_correction2 = 1 - beta2 ** state['step']  # .item()
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # amsgrad = False
                # if amsgrad:
                #     state['max_exp_avg_sqs'] = torch.maximum(state['max_exp_avg_sqs'], exp_avg_sq)
                #     exp_avg_sq = state['max_exp_avg_sqs'] + 0.0

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                # TODO : https://github.com/pytorch/pytorch/issues/32861
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class SAGAdamSimpleBase(SAGBase):
    def __init__(self, params, 
        n, m, batch_mode, init_y_i,
        lr=1e-3, weight_decay=0,
        betas=(0.9, 0.999), eps=1e-8
    ):        
        CustomAdam.check_params(lr, eps, betas)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SAGAdamSimpleBase, self).__init__(params, n, m, batch_mode, init_y_i, lr, weight_decay, defaults=defaults)

        n = self.m if batch_mode else self.n
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0  # torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['grad_sq'] = torch.zeros(n, *p.data.shape, device = p.data.device)

    def step(self, batch_idx, indexes, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1) 
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) 
                
                denom = exp_avg_sq.sqrt()#.add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']  # .item()
                bias_correction2 = 1 - beta2 ** state['step']  # .item()
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                ## SAG
                i = batch_idx if self.batch_mode else indexes
                state['m'][i] = 1.0
                state['grad'][i] = exp_avg + 0.0
                state['grad_sq'][i] = denom + 0.0
        
                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                m = state['m'][i].sum()
                #m = state['m'].sum()
                v = state['grad'].sum(dim=0) / m
                r = (state['grad_sq'].sum(dim=0) / m.sqrt()).add_(group['eps'])
                p.data.addcdiv_(-step_size, v, r)

        return loss

class SAGAdamBase(SAGBase):
    def __init__(self, params, 
        n, m, batch_mode, init_y_i,
        lr=1e-3, weight_decay=0,
        betas=(0.9, 0.999), eps=1e-8
    ):        
        CustomAdam.check_params(lr, eps, betas)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SAGAdamBase, self).__init__(params, n, m, batch_mode, init_y_i, lr, weight_decay, defaults=defaults)

        self.debug = False
        n = self.m if batch_mode else self.n
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0  # torch.zeros(1)
                state['grad_sq'] = torch.zeros(n, *p.data.shape, device = p.data.device)
                if self.debug :
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, batch_idx, indexes, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                state['step'] += 1

                i = batch_idx if self.batch_mode else indexes
                state['m'][i] = 1.0

                state['grad'][i].mul_(beta1).add_(grad, alpha=1-beta1) 
                state['grad_sq'][i].mul_(beta2).addcmul_(grad, grad, value=1-beta2) 

                m = state['m'][i].sum()
                #m = state['m'].sum()
                exp_avg = state['grad'].sum(dim=0) / m
                denom = (state['grad_sq'].sum(dim=0) / m).sqrt().add_(group['eps'])
                            
                bias_correction1 = 1 - beta1 ** state['step']  # .item()
                bias_correction2 = 1 - beta2 ** state['step']  # .item()
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                if not self.debug :
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else :
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1-beta1) 
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) 
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    
                    # print("=========")
                    # print(state['grad_sq'].shape, state['grad_sq'][i].shape)
                    # print(state['grad'].sum(), state['grad_sq'].sum())
                    # print(exp_avg.sum(), mean_y_i.sum())
                    # print(denom.sum(), mean_y_i_sq.sum())
                    

        return loss

class SAGAdamSimpleWithd(SAGWithd):
    def __init__(self, params, 
        n, m, batch_mode, init_y_i,
        lr=1e-3, weight_decay=0,
        betas=(0.9, 0.999), eps=1e-8
    ):        
        CustomAdam.check_params(lr, eps, betas)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SAGAdamSimpleWithd, self).__init__(params, n, m, batch_mode, init_y_i, lr, weight_decay, defaults=defaults)

        n = self.m if batch_mode else self.n
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0  # torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['grad_sq'] = torch.zeros(n, *p.data.shape, device = p.data.device)
                state['d_sq'] = torch.zeros_like(p.data, device = p.data.device)

    def step(self, batch_idx, indexes, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1) 
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) 
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']  # .item()
                bias_correction2 = 1 - beta2 ** state['step']  # .item()
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                ## SAG
                i = batch_idx if self.batch_mode else indexes
                state['m'][i] = 1.0

                m = state['m'][i].sum()
                #m = state['m'].sum()
                if self.batch_mode :
                    y_i = state['grad'][i] / m
                    y_i_sq = state['grad_sq'][i] / m
                else :
                    y_i = state['grad'][i].sum(dim=0) / m
                    y_i_sq = state['grad_sq'][i].sum(dim=0) / m

                state['d'].sub_(y_i, alpha=1.0).add_(exp_avg, alpha=1.0) 
                state['d_sq'].sub_(y_i_sq , alpha=1.0).add_(denom, alpha=1.0) 
                
                state['grad'][i] = exp_avg + 0.0
                state['grad_sq'][i] = denom + 0.0

                exp_avg = state['d']
                denom = state['d_sq'].sqrt().add_(group['eps'])

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                # TODO : https://github.com/pytorch/pytorch/issues/32861
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class SAGAdamWithd(SAGWithd):
    def __init__(self, params, 
        n, m, batch_mode, init_y_i,
        lr=1e-3, weight_decay=0,
        betas=(0.9, 0.999), eps=1e-8
    ):        
        CustomAdam.check_params(lr, eps, betas)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SAGAdamWithd, self).__init__(params, n, m, batch_mode, init_y_i, lr, weight_decay, defaults=defaults)

        self.debug = False
        n = self.m if batch_mode else self.n
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0  # torch.zeros(1)
                state['grad_sq'] = torch.zeros(n, *p.data.shape, device = p.data.device)
                state['d_sq'] = torch.zeros_like(p.data, device = p.data.device)
                if self.debug :
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, batch_idx, indexes, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                state['step'] += 1

                i = batch_idx if self.batch_mode else indexes
                state['m'][i] = 1.0
                m = state['m'][i].sum()
                #m = state['m'].sum()
                if self.batch_mode :
                    y_i = state['grad'][i] / m
                    y_i_sq = state['grad_sq'][i] / m
                else :
                    y_i = state['grad'][i].sum(dim=0) / m
                    y_i_sq = state['grad_sq'][i].sum(dim=0) / m

                buf    = beta1 * y_i    + (1-beta1) * grad
                buf_sq = beta2 * y_i_sq + (1-beta2) * grad * grad

                state['d'].sub_(y_i, alpha=1.0).add_(buf, alpha=1.0)
                state['d_sq'].sub_(y_i_sq, alpha=1.0).add_(buf_sq, alpha=1.0)

                state['grad'][i] = buf
                state['grad_sq'][i] = buf_sq

                exp_avg = state['d']
                denom = state['d_sq'].sqrt().add_(group['eps'])

                #    
                bias_correction1 = 1 - beta1 ** state['step']  # .item()
                bias_correction2 = 1 - beta2 ** state['step']  # .item()
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                if self.debug :
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1-beta1) 
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) 
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.data.addcdiv_(-step_size, exp_avg, denom)
                    
                # print("=========")
                # #print(state['grad_sq'].shape, state['grad_sq'][i].shape)
                # #print(state['grad'].sum(), state['grad_sq'].sum())
                # print(exp_avg.sum())#, mean_y_i.sum())
                # print(denom.sum())#, mean_y_i_sq.sum())
                    
        return loss

class AdamInverseSqrtWithWarmup(CustomAdam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7,
                 exp_factor=0.5):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.exp_factor = exp_factor
        self.decay_factor = warmup_end_lr * warmup_updates ** self.exp_factor

        # total number of updates
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -self.exp_factor)

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(param_group['num_updates'])

class AdamCosineWithWarmup(CustomAdam):
    """
    Assign LR based on a cyclical schedule that follows the cosine function.
    See https://arxiv.org/pdf/1608.03983.pdf for details.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``).
    During warmup::
        lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
        lr = lrs[update_num]
    After warmup::
        lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(t_curr / t_i))
    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7,
                 min_lr=1e-9, init_period=1000000, period_mult=1, lr_shrink=0.75):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, apply cosine scheduler
        self.min_lr = min_lr
        self.max_lr = lr
        self.period = init_period
        self.period_mult = period_mult
        self.lr_shrink = lr_shrink

        # total number of updates
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            t = num_updates - self.warmup_updates
            if self.period_mult == 1:
                pid = math.floor(t / self.period)
                t_i = self.period
                t_curr = t - (self.period * pid)
            else:
                pid = math.floor(math.log(1 - t / self.period * (1 - self.period_mult), self.period_mult))
                t_i = self.period * (self.period_mult ** pid)
                t_curr = t - (1 - self.period_mult ** pid) / (1 - self.period_mult) * self.period
            lr_shrink = self.lr_shrink ** pid
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_curr / t_i))

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(param_group['num_updates'])

