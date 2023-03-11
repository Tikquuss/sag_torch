import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from src.optim import get_optimizer
from src.datasets.scm import get_dataloader, get_weights, forward, g_dic

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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    s={"N":100, "M":5, "K":2, "fixed_w":False, "scm":False, "g":"id", "noise":2.0, "out_dim":1}
    #s={"N":4, "M":4, "K":4, "fixed_w":False, "scm":False, "g":"id", "noise":0.0, "out_dim":1}
    train_size, val_size = 10, 100
    k = [1, 50, 100000]
    k = None
    train_batch_size, val_batch_size = 1024, 1024
    max_epoch = 10000
    lr = 0.2
    optim_name="adam"
    all_optims = get_all_optims(weight_decay=0.1, momentum=0.9, beta1=0.9, beta2=0.99)

    mu_w, sigma_w = 0.0, 1.0 # feature map
    mu_v, sigma_v = 0.0, 1.0 # output layer
    N, M, K, fixed_w, scm, g = s["N"], s["M"], s["K"], s["fixed_w"], s["scm"], s["g"] 
    noise, out_dim = s.get("noise", 0.0), s.get("out_dim", 1)
    act_funct = g_dic[g]
    if g in ["relu", 'id'] : sigma_w = sigma_v = N**-0.5
    train_loader, val_loader, data_infos = get_dataloader(
        train_size, val_size, N, M, k=k, out_dim = out_dim, g=act_funct, 
        mu_x = 0.0, sigma_x = 1.0, # data
        mu_w = mu_w, sigma_w = sigma_w, # feature map
        mu_v = mu_v, sigma_v = sigma_v, # output layer
        mu_noise = 0.0, sigma_noise = noise, # noise
        seed = 100, task = "regression", 
        include_indexes = False, 
        train_batch_size = train_batch_size, val_batch_size = val_batch_size, 
        num_workers=0, 
        return_just_set = False
    )
    for key, val in data_infos.items()  : print(str(key) + " --> " + str(val))

    w, v = get_weights(N, M=K, out_dim = out_dim,
            mu_w = 0.0, sigma_w = 1.0, # feature map
            mu_v = 0.0, sigma_v = 1.0, # output layer
    )
    w.requires_grad_(True)
    v.requires_grad_(True)
    w = w.to(device)
    v = v.to(device)
    w=torch.nn.Parameter(w)
    v=torch.nn.Parameter(v)
    if fixed_w : 
        w.requires_grad = False
    if scm : 
        v = torch.ones_like(v, device=device)
        v.requires_grad = False

    params = [{'params':[w], 'lr':lr}, {'params':[v], 'lr':lr}]
    optimizer = get_optimizer(params,  all_optims[optim_name] + f",lr={lr}")

    train_losses, val_losses = [], []
    for ep in tqdm.tqdm(range(max_epoch)) :
        train_loss, val_loss = [], []
        for x, y in train_loader :
            x, y = x.to(device), y.to(device)
            y_hat, _, _ = forward(w, v, x, N, g=act_funct) 
            loss = F.mse_loss(input=y_hat, target=y, size_average=None, reduce=None, reduction='mean')
            # loss.backward()
            # if not fixed_w : 
            #     w.data = w.data - lr * w.grad
            #     w.grad.zero_()
            # if not scm : 
            #     v.data = v.data - lr * v.grad
            #     v.grad.zero_()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        with torch.no_grad():
            for x, y in val_loader :
                x, y = x.to(device), y.to(device)
                y_hat, _, _ = forward(w, v, x, N, g=act_funct) 
                loss = F.mse_loss(input=y_hat, target=y, size_average=None, reduce=None, reduction='mean')
                val_loss.append(loss.item())
        
        train_losses.append(sum(train_loss)/len(train_loss))
        val_losses.append(sum(val_loss)/len(val_loss))

    L, C = 1, 1
    #figsize=(C*15, L*10)
    figsize=(C*6, L*4)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(L, C, 1)
    ax.plot(train_losses, label="train")
    ax.plot(val_losses,   label="val")
    ax.axhline(y = 0.0, color = "red", linestyle = '--')
    ax.set_xlabel('epoch')
    ax.set_ylabel("loss")
    ax.set_xscale("log")
    ax.legend()
    plt.show()