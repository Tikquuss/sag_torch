
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

class SAGSGDBase(SAGBase):
    def __init__(self, params, 
        n, m, batch_mode, init_y_i,
        lr=1e-3, weight_decay=0, sum_all = True,
        momentum=0, dampening=0, nesterov=False
    ):        
        CustomSGD.check_params(lr, momentum, weight_decay, nesterov, dampening)
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SAGSGDBase, self).__init__(params, n, m, batch_mode, init_y_i, lr, weight_decay, sum_all = sum_all, defaults=defaults)

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
                i = batch_idx if self.batch_mode else indexes
                state['m'][i] = 1.0
                state['grad'][i] = grad + 0.0
                m = state['m'].sum() if self.sum_all else state['m'][i].sum()

                p.data.add_(-group['lr'], state['grad'].sum(dim=0) / m)

        return loss

class SAGSGDWithd(SAGBase):
    def __init__(self, params, 
        n, m, batch_mode, init_y_i,
        lr=1e-3, weight_decay=0, sum_all = True,
        momentum=0, dampening=0, nesterov=False
    ):        
        CustomSGD.check_params(lr, momentum, weight_decay, nesterov, dampening)
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SAGSGDWithd, self).__init__(params, n, m, batch_mode, init_y_i, lr, weight_decay, sum_all = sum_all, defaults=defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['d'] = torch.zeros_like(p.data, device = p.data.device)

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
                i = batch_idx if self.batch_mode else indexes
                state['m'][i] = 1.0
                m = state['m'].sum() if self.sum_all else state['m'][i].sum()
                if self.batch_mode : y_i = state['grad'][i] / m
                else : y_i = state['grad'][i].sum(dim=0) / m
                state['d'].sub_(y_i, alpha=1.0).add_(grad, alpha=1.0) 
                state['grad'][i] = grad + 0.0

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                p.data.add_(-group['lr'], state['d'])
                
        return loss