
import torch
from torch import optim

from .sag import SAGBase, SAGWithd

class CustomSGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):

        self.check_params(lr, momentum, weight_decay, nesterov, dampening)

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        super(CustomSGD, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)

        self.add_weight_decay_before = True

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @staticmethod
    def check_params(lr, momentum, weight_decay, nesterov, dampening):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

    def step(self, closure=None):
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
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse: pass
                state = self.state[p]
                
                if self.add_weight_decay_before :
                    if group['weight_decay'] != 0:
                        #grad = grad.add(p.data, alpha=group['weight_decay'])
                        grad.add_(group['weight_decay'], p.data)

                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - group['dampening'], grad)
                    
                    # nesterov
                    if group['nesterov'] :
                        grad = grad.add(momentum, buf)
                    else:
                        grad = buf                
                
                if not self.add_weight_decay_before :
                    if group['weight_decay'] != 0:
                        p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                p.data.add_(-group['lr'], grad)

        return loss

class SAGSGD(SAGBase):
    def __init__(self, params, 
        n, m, batch_mode, init_y_i,
        lr=1e-3, weight_decay=0,
        momentum=0, dampening=0, nesterov=False
    ):        
        CustomSGD.check_params(lr, momentum, weight_decay, nesterov, dampening)
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SAGSGD, self).__init__(params, n, m, batch_mode, init_y_i, lr, weight_decay, defaults=defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)

        self.add_weight_decay_before = False

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse: pass

                state = self.state[p]
                
                ## SGD
                if self.add_weight_decay_before :
                    if group['weight_decay'] != 0:
                        #grad = grad.add(p.data, alpha=group['weight_decay'])
                        grad.add_(group['weight_decay'], p.data)

                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - group['dampening'], grad)
                    
                    # nesterov
                    if group['nesterov'] :
                        grad = grad.add(momentum, buf)
                    else:
                        grad = buf                
                
                if not self.add_weight_decay_before :
                    if group['weight_decay'] != 0:
                        p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                #p.data.add_(-group['lr'], grad)

                ## SAG
                if self.batch_mode :
                    state['grad'][batch_idx] = grad + 0.0
                    state['m'][batch_idx] = 1.0
                    #mean_y_i = state['grad'].sum(dim=0) / self.m
                    mean_y_i = state['grad'].sum(dim=0) / state['m'].sum()
                else :
                    state['grad'][indexes] = grad + 0.0
                    state['m'][indexes] = 1.0
                    #mean_y_i = state['grad'].sum(dim=0) / self.n
                    mean_y_i = state['grad'].sum(dim=0) / state['m'].sum()

                p.data.add_(-group['lr'], mean_y_i)

        return loss