import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os

from src.optim import get_optimizer
from src.datasets.scm import get_dataloader, get_weights, forward, iid_normal, \
      iid_mixture_normal, get_modulation_matrix_multi_singular, binarize
from src.dataset import g_dic
from src.optim import get_all_optims
from src.utils import AttrDict

def get_acc(logits, y, task):
    if "multi" in task :
        # logits : (P, n_class) 
        #return (F.softmax(logits, dim=-1).argmax(dim=-1) == y).float().mean().item()
        return (logits.argmax(dim=-1) == y).float().mean().item()
    elif "bin" in task : 
        # logits : (P,) 
        return (torch.sigmoid(logits).round() == y).float().mean().item()
    
def one_epoch_finite_data(w, v, train_loader, val_loader, criterion, optimizer, h_params, act_funct, denom, device):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    N, K, out_dim = h_params.N, h_params.K, h_params.out_dim
    for x, y in train_loader :
        x, y = x.to(device), y.to(device)
        logits, _, _ = forward(w, v, x, N, g=act_funct, denom=denom)
        loss = criterion(input=logits, target=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ##
        epsilon_w = h_params.noise_optim*iid_normal(dim=N, sample_shape=(K,), mu=0.0, sigma=h_params.inv_tmp) # (K, N)
        w.data = w.data + epsilon_w.to(device)
        epsilon_v = h_params.noise_optim*iid_normal(dim=out_dim, sample_shape=(K,), mu=0.0, sigma=h_params.inv_tmp) # (K, out_dim)
        v.data = v.data + epsilon_v.to(device)
        ##
        train_loss.append(loss.item())
        if "classification" in h_params.task : train_acc.append(get_acc(logits, y, h_params.task))

        with torch.no_grad():
            for x, y in val_loader :
                x, y = x.to(device), y.to(device)
                logits, _, _ = forward(w, v, x, N, g=act_funct, denom=denom) 
                loss = criterion(input = logits, target=y)
                val_loss.append(loss.item())
                if "classification" in h_params.task : val_acc.append(get_acc(logits, y, h_params.task))

    return train_loss, val_loss, train_acc, val_acc

def one_epoch_infinite_data(w, v, w_start, v_start, criterion, optimizer, h_params, act_funct, denom, device):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    N, K, out_dim = h_params.N, h_params.K, h_params.out_dim
    P = h_params.train_size
    if h_params.weights_x is None : x = iid_normal(dim=N, sample_shape=(P,), mu=h_params.mu_x, sigma=h_params.sigma_x) # (P, N)
    else : x = iid_mixture_normal(dim=N, sample_shape=(P,), mu=h_params.mu_x, sigma=h_params.sigma_x, weights=h_params.weights_x) # (P, N)
    noise_vec = iid_normal(dim=out_dim, sample_shape=(P,), mu=0.0, sigma=0.1) # (P, out_dim)
    F_matrix = torch.eye(N)
    if h_params.k is not None :
        F_matrix, _, _, _ = get_modulation_matrix_multi_singular(N, h_params.k, h_params.singular_val) # (N, N)
        F_matrix = torch.from_numpy(F_matrix).float()
    x = x @ F_matrix # (P, N)
    x = x.to(device)
    F_matrix = F_matrix.to(device)
    y, _, _ = forward(w_start, v_start, x, N, g=act_funct) # (P, out_dim)
    sigma_noise = h_params.noise
    y = y + sigma_noise*noise_vec[:len(y)].to(device)
    if "bin" in h_params.task : y = binarize(y)
    if "multi" in h_params.task : pass
    logits, _, _ = forward(w, v, x, h_params.N, g=act_funct, denom=denom)
    loss = criterion(input=logits, target=y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ##
    epsilon_w = h_params.noise_optim*iid_normal(dim=N, sample_shape=(K,), mu=0.0, sigma=h_params.inv_tmp) # (K, N)
    w.data = w.data + epsilon_w.to(device)
    epsilon_v = h_params.noise_optim*iid_normal(dim=out_dim, sample_shape=(K,), mu=0.0, sigma=h_params.inv_tmp) # (K, out_dim)
    v.data = v.data + epsilon_v.to(device)
    ##
    train_loss.append(loss.item())
    if "classification" in h_params.task : train_acc.append(get_acc(logits, y, h_params.task))
    #val_loss.append(train_loss[-1])
    #if "classification" in h_params.task : val_acc.append(train_acc[-1])
    with torch.no_grad():
        P = h_params.val_size
        if h_params.weights_x is None : x = iid_normal(dim=N, sample_shape=(P,), mu=h_params.mu_x, sigma=h_params.sigma_x) # (P, N)
        else : x = iid_mixture_normal(dim=N, sample_shape=(P,), mu=h_params.mu_x, sigma=h_params.sigma_x, weights=h_params.weights_x) # (P, N)
        noise_vec = iid_normal(dim=out_dim, sample_shape=(P,), mu=0.0, sigma=0.1) # (P, out_dim)
        x = x.to(device)
        y, _, _ = forward(w_start, v_start, x, N, g=act_funct) # (P, out_dim)
        # test data is noiseless
        #y = y + 0.0*noise_vec[:len(y)].to(device)
        logits, _, _ = forward(w, v, x, N, g=act_funct, denom=denom) 
        loss = criterion(input = logits, target=y)
        val_loss.append(loss.item())
        if "classification" in h_params.task : val_acc.append(get_acc(logits, y, h_params.task))

    return train_loss, val_loss, train_acc, val_acc


def plot_and_fill_between(ax, dict_v, min_ep, max_ep, label) :
    x_, y_  = list(dict_v[0].keys()), list(dict_v[0].values())
    x_ = x_[min_ep:max_ep]
    y_ = y_[min_ep:max_ep]
    #x_ = [i/N for i in x_]
    ax.plot(x_, y_, label=label)
    if dict_v[1] :
        std = list(dict_v[1].values())[min_ep:max_ep]
        y_min = [y_[i] - std[i] for i in range(len(y_))]
        y_max = [y_[i] + std[i] for i in range(len(y_))]
        ax.fill_between(x_, y_min, y_max, alpha=0.07, interpolate=True)

def do_plot(h_params, 
            train_losses, val_losses, train_accuracies, val_accuracies, 
            Qs = None, Rs = None, Ts = None,
            tl_std = None, vl_std = None, ta_std = None, va_std = None,
            Qs_std = None, Rs_std = None, Ts_std = None
            ):

    min_ep, max_ep = 0, h_params.max_epoch*2
    L, C = (2 if "classification" in h_params.task else 1), 2
    if Qs : L += 1
    #figsize=(C*15, L*10)
    figsize=(C*6, L*4)
    fig = plt.figure(figsize=figsize)
    for i, train_m, val_m, label, threshold in zip(
        [0] + ([2] if "classification" in h_params.task else []),
        [(train_losses, tl_std), (train_accuracies, ta_std)],
        [(val_losses, vl_std), (val_accuracies, va_std)],
        ['loss', "accuracy"],
        [0.0, 1.0]
    ):
        for j in [1, 2] :
            ax = fig.add_subplot(L, C, i+j)
            plot_and_fill_between(ax, train_m, min_ep, max_ep, label="train")
            if val_m : plot_and_fill_between(ax, val_m, min_ep, max_ep, label="val")
            ax.axhline(y = threshold, color = "red", linestyle = '--')
            ax.set_xlabel('epoch')
            ax.set_ylabel(label)
            ax.legend()
        ax.set_xscale("log")
        ax.set_xlabel('epoch (log scale)')
        #ax.set_yscale("log")

    if Qs :
        for k in [i+j+1, i+j+2] :
            ax = fig.add_subplot(L, C, k)
            plot_and_fill_between(ax, (Rs, Rs_std), min_ep, max_ep, label="Rs")
            plot_and_fill_between(ax, (Qs, Qs_std), min_ep, max_ep, label="Qs")
            plot_and_fill_between(ax, (Ts, Ts_std), min_ep, max_ep, label="Ts")
            #ax.plot(e_g, label="e_g")
            ax.set_xlabel('epoch')
            ax.set_ylabel("alignment")
            ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel('epoch (log scale)')

    # save the image  
    if h_params.file_name :
        os.makedirs(h_params.save_path, exist_ok=True)
        plt.savefig(os.path.join(h_params.save_path, f'{h_params.file_name}.png'), bbox_inches='tight')
        #plt.savefig(os.path.join(h_params.save_path, f'{file_name}.pdf'), bbox_inches='tight', format = "pdf")
    if h_params.show_plot : plt.show()
    
def train(h_params, plot = True) :

    if h_params.seed :
        np.random.seed(h_params.seed)
        torch.manual_seed(h_params.seed)  

    device = torch.device(h_params.device if torch.cuda.is_available() else h_params.device)

    N, M, K, fixed_w, scm, g = h_params.N, h_params.M, h_params.K, h_params.fixed_w, h_params.scm, h_params.g 
    noise, out_dim = getattr(h_params, "noise", 0.0), getattr(h_params, "out_dim", 1)
    if h_params.task == "classification" : h_params.task = f"{'bin' if out_dim==1 else 'multi'}_{h_params.task}"
    act_funct = g_dic[h_params.g]
    if h_params.g in ["relu", 'id'] : h_params.sigma_w = h_params.sigma_v = h_params.N**-0.5
    if h_params.K==1 and h_params.scm : denom = 1.0
    else : denom = np.sqrt(h_params.N)
    train_loader, val_loader, data_infos, w_start, v_start, F_matrix = get_dataloader(
        h_params.train_size, h_params.val_size, N, M, 
        k=h_params.k, singular_val=h_params.singular_val,
        out_dim = out_dim, g=act_funct, 
        mu_x = h_params.mu_x, sigma_x = h_params.sigma_x, weights_x = h_params.weights_x, # data
        mu_w = h_params.mu_w, sigma_w = h_params.sigma_w, # feature map
        mu_v = h_params.mu_v, sigma_v = h_params.sigma_v, # output layer
        mu_noise = 0.0, sigma_noise = noise, # noise
        scm = scm,
        seed = h_params.seed, task = h_params.task, 
        include_indexes = "sag" in h_params.optim_name, 
        train_batch_size = h_params.train_batch_size, 
        val_batch_size = h_params.val_batch_size, 
        num_workers = 0, 
        return_just_set = False
    )

    for key, val in data_infos.items()  : print(str(key) + " --> " + str(val))
    w_start = w_start.to(device)
    v_start = v_start.to(device)
    F_matrix = F_matrix.to(device)

    w, v = get_weights(N, M=K, out_dim = out_dim,
            mu_w = h_params.mu_w, sigma_w = h_params.sigma_w, # feature map
            mu_v = h_params.mu_v, sigma_v = h_params.sigma_v, # output layer
    )
    w.requires_grad_(True)
    v.requires_grad_(True)
    w = w.to(device)#.float()
    v = v.to(device)#.float()
    w=torch.nn.Parameter(w)
    v=torch.nn.Parameter(v)
    if fixed_w : w.requires_grad = False
    if scm : v = torch.ones_like(v, device=device, requires_grad=False)*h_params.sigma_v

    all_optims = get_all_optims(
        weight_decay=h_params.weight_decay, momentum=h_params.momentum, 
        beta1=h_params.beta1, beta2=h_params.beta2
    )
    params = [{'params':[w], 'lr':h_params.lr}, {'params':[v], 'lr':h_params.lr}]
    optimizer = get_optimizer(params,  all_optims[h_params.optim_name] + f",lr={h_params.lr}")

    if h_params.task == "regression" : criterion = nn.MSELoss()
    else : 
        if "multi" in h_params.task : criterion = nn.CrossEntropyLoss() 
        elif "bin" in h_params.task : criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = {}, {}
    train_accuracies, val_accuracies = {}, {}
    Rs, Qs, Ts, e_g = {}, {}, {}, {}
    i, j = 0, 0
    T = (1/N)*(w_start.data @ F_matrix.T @ F_matrix @ w_start.data.T) # (M, N) x (N, N) x (N, N) x (N, M)
    T = T.cpu().numpy()
    try :
        for ep in (pbar := tqdm.tqdm(range(1, h_params.max_epoch+1))):
            ##########################################
            ##########################################
            train_loss, val_loss, train_acc, val_acc = one_epoch_finite_data(
                w, v, train_loader, val_loader, criterion, optimizer, h_params, act_funct, denom, device
            )
            ##########################################
            ##########################################
            # train_loss, val_loss, train_acc, val_acc = one_epoch_infinite_data(
            #     w, v, w_start, v_start, criterion, optimizer, h_params, act_funct, denom, device
            # )
            ##########################################
            ##########################################

            train_losses[ep] = sum(train_loss)/len(train_loss)
            val_losses[ep] = sum(val_loss)/len(val_loss)
            pbar_infos = f"train_loss : {round(train_losses[ep],5)}, val_loss : {round(val_losses[ep],5)}"

            if "classification" in h_params.task :
                train_accuracies[ep] = sum(train_acc)/len(train_acc)
                val_accuracies[ep] = sum(val_acc)/len(val_acc)
                pbar_infos += f", train_acc : {round(train_accuracies[ep],5)}, val_acc : {round(val_accuracies[ep],5)}"
            
            pbar.set_description(pbar_infos)
            R = (1/N)*(w_start @ F_matrix @ w.data.T) # (M, N) x (N, N) x (N, K) = (M, K)
            Q = (1/N)*(w.data @ F_matrix.T @ F_matrix @ w.data.T) # (K, N) x (N, N) x (N, N) x (N, K) = (K, K)
            Rs[ep] = R.cpu().numpy()[i][j]
            Qs[ep] = Q.cpu().numpy()[i][j]
            Ts[ep] = T[i][j]
            # e_g[ep] = (1/2)*(1+Q-2*R)

            if val_losses and val_losses[ep] < h_params.loss_tolerance : break
    except KeyboardInterrupt:
        pass

    if plot :
        if h_params.file_name is None : 
            h_params.file_name = f'{N}_{g}_{K}_{h_params.train_size}_{h_params.val_size}_{h_params.task}_{h_params.optim_name}_{h_params.max_epoch}'
        do_plot(h_params, train_losses, val_losses, train_accuracies, val_accuracies, Qs, Rs, Ts)

    return train_losses, val_losses, train_accuracies, val_accuracies, Rs, Qs, Ts

mean_function = lambda p : sum(p)/len(p)
std_function  = lambda p, mu : (sum([(x-mu)**2 for x in p])/len(p))**0.5
def mu_and_std(p):
    mu = mean_function(p)
    return mu, std_function(p, mu)

def mu_and_std_for_dict(p_dict, min_ep):
    mean_dict = {}
    std_dict = {}
    for ep in range(1, min_ep) :
        p = [ta[ep] for ta in p_dict]
        mean_dict[ep] = mean_function(p)
        std_dict[ep] = std_function(p, mu=mean_dict[ep])
    return mean_dict, std_dict

def train_mutiple_seeds(h_params, plot = True) :
    seeds = h_params.seed
    if (seeds is None) or type(seeds) == int or len(seeds) == 1 :
        return train(h_params, plot)
        

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    Rs, Qs, Ts = [0 for _ in seeds], [0 for _ in seeds], [0 for _ in seeds]

    min_ep = np.inf
    for i, s in enumerate(seeds) :
        h_params.seed = s
        tl, vl, ta, va, Rs[i], Qs[i], Ts[i] = train(h_params, plot=False)
        train_losses.append(tl)
        val_losses.append(vl)
        train_accuracies.append(ta)
        val_accuracies.append(va)
        min_ep = min(min_ep, len(tl))

    train_losses, tl_std = mu_and_std_for_dict(train_losses, min_ep)
    val_losses,   vl_std = mu_and_std_for_dict(val_losses, min_ep)
    if "classification" in h_params.task :
        train_accuracies, ta_std = mu_and_std_for_dict(train_accuracies, min_ep)
        val_accuracies,   va_std = mu_and_std_for_dict(val_accuracies, min_ep)
    else : ta_std, va_std = None, None

    if Qs :
        Qs, Qs_std = mu_and_std_for_dict(Qs, min_ep)
        Rs, Rs_std = mu_and_std_for_dict(Rs, min_ep)
        Ts, Ts_std = mu_and_std_for_dict(Ts, min_ep)

    if h_params.file_name is None : 
        h_params.file_name = f'{h_params.N}_{h_params.g}_{h_params.K}_{h_params.train_size}_{h_params.val_size}_{h_params.task}_{h_params.optim_name}_{h_params.max_epoch}'
    
    do_plot(h_params, train_losses, val_losses, train_accuracies, val_accuracies, 
            Qs, Rs, Ts,
            tl_std, vl_std, ta_std, va_std, Qs_std, Rs_std, Ts_std)
    
    return train_losses, val_losses, train_accuracies, val_accuracies, Qs, Rs, Ts, \
            tl_std, vl_std, ta_std, va_std, Qs_std, Rs_std, Ts_std


def sweep(h_params, list_v):
    pass
    
if __name__ == "__main__":
    
    seed=list(range(1))
    seed=None

    train_size, val_size = 10, 1000
    #max_epoch = 1000*100
    max_epoch = 5000

    k = [10, 1]
    k = {10 : 0.2}
    k = None
    
    h_params = {
        ###### Model ######
        "N":10, "M":1, "K":1, "out_dim":1,
        "fixed_w":False, "scm":True, "g":"sigmoid", 
        
        # feature map
        "mu_w" : 0.0, "sigma_w" : 1.0, 
        # output layer
        "mu_v": 0.0, "sigma_v" : 1.0, 

        ###### Data ###### 
        "noise":0.001, 
        "train_size" : train_size, "val_size" : val_size,
        "train_batch_size" : 2**20, "val_batch_size" : 2**20,
        #"mu_x" : 0.0, "sigma_x" : 1.0, "weights_x" : None,
        "mu_x" : [-1.0, 0.0, 1.0], "sigma_x" : [4.0, 1.0, 4.0], "weights_x" : [0.3, 0.4, 0.3],
        "k" : k, "singular_val" : 1.0,

        ###### Training ###### 
        "seed" : seed,
        "task" : "regression",
        #"task" : "classification",
        "max_epoch" : max_epoch,
        "device" : 'cuda',
        'loss_tolerance' : 0.0001,

        ###### Optimizer ######
        # "optim_name":"custom_adam",
        "optim_name":"sgd",
        "lr" : 0.2,
        "weight_decay": 0.0, 
        "momentum":0.9, 
        "beta1":0.9, 
        "beta2":0.99,
        "noise_optim" : 0.0, 
        "inv_tmp": 2/10000000,

        ###### Plot ######
        "save_path" : f"D:/Canada/MILA/2023_path_to_PhD/double descent/images_scm",
        "file_name" : None,
        "show_plot" : True,
    }

    h_params = AttrDict(h_params)
    #train(h_params, plot = True)
    train_mutiple_seeds(h_params, plot = True)