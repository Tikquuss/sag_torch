import torch

class SimpleSAG(torch.optim.Optimizer):
    """
    Stochastic Average Gradient (SAG) : vanilla implementation
    """
    def __init__(self, params, n, m, batch_mode, init_y_i, lr=1e-3, weight_decay=0):
        if not 0.0 <= lr: raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.batch_mode = batch_mode
        self.n = int(m) if batch_mode else int(n)
        self.init_y_i = init_y_i

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grad'] = torch.zeros_like(p.data)
                state['d'] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, batch_idx, indexes, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse: pass # TODO : sparse SAG
                state = self.state[p]
                state['d'].sub_(state['grad'], alpha=1.0).add_(grad, alpha=1.0) 
                state['grad'] = grad + 0.0
                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)
                p.data.add_(-group['lr'], state['d'])

        return loss

class SAGBase(torch.optim.Optimizer):
    """
    Stochastic Average Gradient (SAG) : vanilla implementation
    """
    def __init__(self, params, n, m, batch_mode, init_y_i, lr=1e-3, weight_decay=0, sum_all = True, defaults = {}):
        if not 0.0 <= lr : raise ValueError("Invalid learning rate: {}".format(lr))
        def_tmp = dict(lr=lr, weight_decay=weight_decay)
        defaults = {**def_tmp, **defaults}
        super().__init__(params, defaults)
        self.batch_mode = batch_mode
        self.n = int(n)
        self.m = int(m)
        self.init_y_i = init_y_i
        self.sum_all = sum_all
        
        n = self.m if batch_mode else self.n
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grad'] = torch.zeros(n, *p.data.shape, device = p.data.device)
                state['m'] = torch.zeros(n)

    def __setstate__(self, state):
        super().__setstate__(state)

    def init_grad(self, device, optimizer = None, train_loader=None, model=None):
        """init y_i = f'_i - g' """
        assert (optimizer is not None) ^ (train_loader is not None)

        if optimizer is not None :
            for group, group2 in zip(self.param_groups, optimizer.param_groups):
                    for p, p2 in zip(group['params'], group2['params']):
                        if p.grad is None: continue
                        self.state[p]['grad'] = optimizer.state[p2]['grad']
            return 

        assert model is not None
        batch_mode = self.batch_mode
        #  delta_f (y_i) & delta_g
        """
        Since $\nabla g = \frac{1}{n} \sum_{i=1}^n \nabla f_i$ with potentially high $n$, 
        it would be impractical to calculate all $\nabla f_i$'s before averaging. We proceed iteratively as follows.

        If we want to calculate $x = \frac{1}{n} \sum_{i=1}^n x_i$ for a given sequence $\{x_1, \dots, x_n\}$, we can 
        set $a_k = \frac{1}{k} \sum_{i=1}^n x_i$ and notice that $a_k = \frac{k-1}{k} a_{k-1} + \frac{1}{k} x_k$. 
        This allows us to compute $a_k$ iteratively up to rank $n$ and set $x = a_n$.
        More precisely, 
            - $x = 0$, 
            - and for $k$ in $\{1, ...., n\}$ : 
                - $x = \frac{k-1}{k} x + \frac{1}{k} x_k$.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['delta_g'] = torch.zeros_like(p.data, device = p.data.device)
        #
        k = 0
        for batch_idx, (data, target, indexes) in enumerate(train_loader):
            loss, _, _, _ = model._get_loss(batch=(data.to(device), target.to(device), indexes))
            self.zero_grad()
            loss.backward()
            k+=1
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    grad = p.grad.data
                    state = self.state[p]
                    state['grad'][batch_idx if batch_mode else indexes] = grad + 0.0
                    state['delta_g'].add_(state['delta_g'], alpha = (k-1.0)/k).add_(grad, alpha = 1.0/k)
        #
        for batch_idx, (data, target, indexes) in enumerate(train_loader):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    state = self.state[p]
                    state['grad'][batch_idx if batch_mode else indexes].sub_(state['delta_g']) 
        #
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                #self.state[p].pop('delta_g', None)
                del self.state[p]['delta_g']

    def step(self, batch_idx, indexes, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse: pass # TODO : sparse SAG
                state = self.state[p]
                
                i = batch_idx if self.batch_mode else indexes
                state['m'][i] = 1.0
                state['grad'][i] = grad + 0.0

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                #m = self.m
                m = state['m'].sum() if self.sum_all else state['m'][i].sum()
                p.data.add_(-group['lr'], state['grad'].sum(dim=0) / m)

                # print("=========")
                # print(state['grad'].shape, state['grad'].sum())

        return loss

class SAGWithd(SAGBase):
    """
    Stochastic Average Gradient (SAG) : vanilla implementation
    """
    def __init__(self, params, n, m, batch_mode, init_y_i, lr=1e-3, weight_decay=0, sum_all = True, defaults = {}):
        super().__init__(params, n, m, batch_mode, init_y_i, lr, weight_decay, sum_all, defaults = defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['d'] = torch.zeros_like(p.data, device = p.data.device)
    
    def step(self, batch_idx, indexes, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse: pass # TODO : sparse SAG
                
                state = self.state[p]
                
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

                # print("=========")
                # print(state['grad'].shape, state['grad'].sum())

        return loss