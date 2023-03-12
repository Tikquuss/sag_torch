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

def get_acc(logits, y, task):
    if "multi" in task :
        # logits : (P, n_class) 
        #return (F.softmax(logits, dim=-1).argmax(dim=-1) == y).float().mean().item()
        return (logits.argmax(dim=-1) == y).float().mean().item()
    elif "bin" in task : 
        # logits : (P,) 
        return (torch.sigmoid(logits).round() == y).float().mean().item()

if __name__ == "__main__":
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seed=100
    #np.random.seed(seed)
    #torch.manual_seed(seed)

    s={"N":20, "M":1, "K":1, "fixed_w":False, "scm":False, "g":"sigmoid", "noise":0.00, "out_dim":1}
    #s={"N":4, "M":4, "K":4, "fixed_w":False, "scm":False, "g":"id", "noise":0.0, "out_dim":1}
    train_size, val_size = 20, 1000
    mu_x, sigma_x, weights_x = 0.0, 1.0, None
    #mu_x, sigma_x, weights_x = [-1.0, 1.0], [4.0, 4.0], [0.5, 0.5]

    k = [10, 1]
    k = {10 : 0.2}
    k = None
    singular_val=1.0
    
    max_epoch = 1000*100
    noise_optim, inv_tmp = 0.0, 2/10000000
    lr = 0.2
    optim_name="custom_adam"
    optim_name="sgd"
    all_optims = get_all_optims(weight_decay=0.0, momentum=0.9, beta1=0.9, beta2=0.99)

    task = "regression"
    #task = "classification" 
    
    train_batch_size, val_batch_size = 2**20, 2**20
    mu_w, sigma_w = 0.0, 1.0 # feature map
    mu_v, sigma_v = 0.0, 0.1 # output layer
    N, M, K, fixed_w, scm, g = s["N"], s["M"], s["K"], s["fixed_w"], s["scm"], s["g"] 
    noise, out_dim = s.get("noise", 0.0), s.get("out_dim", 1)
    if task == "classification" : task = f"{'bin' if out_dim==1 else 'multi'}_{task}"
    act_funct = g_dic[g]
    if g in ["relu", 'id'] : sigma_w = sigma_v = N**-0.5
    if K==1 and scm : denom = 1.0
    else : denom = np.sqrt(N)
    train_loader, val_loader, data_infos, w_start, v_start, F_matrix = get_dataloader(
        train_size, val_size, N, M, 
        k=k, singular_val=singular_val,
        out_dim = out_dim, g=act_funct, 
        mu_x = mu_x, sigma_x = sigma_x, weights_x=weights_x, # data
        mu_w = mu_w, sigma_w = sigma_w, # feature map
        mu_v = mu_v, sigma_v = sigma_v, # output layer
        mu_noise = 0.0, sigma_noise = noise, # noise
        scm = scm,
        seed = seed, task = task, 
        include_indexes = False, 
        train_batch_size = train_batch_size, val_batch_size = val_batch_size, 
        num_workers=0, 
        return_just_set = False
    )
    for key, val in data_infos.items()  : print(str(key) + " --> " + str(val))
    w_start = w_start.to(device)
    v_start = v_start.to(device)
    F_matrix = F_matrix.to(device)

    w, v = get_weights(N, M=K, out_dim = out_dim,
            mu_w = mu_w, sigma_w = sigma_w, # feature map
            mu_v = mu_v, sigma_v = sigma_v, # output layer
    )
    w.requires_grad_(True)
    v.requires_grad_(True)
    w = w.to(device)#.float()
    v = v.to(device)#.float()
    w=torch.nn.Parameter(w)
    v=torch.nn.Parameter(v)
    if fixed_w : 
        w.requires_grad = False
    if scm : 
        v = torch.ones_like(v, device=device)
        v.requires_grad = False

    params = [{'params':[w], 'lr':lr}, {'params':[v], 'lr':lr}]
    optimizer = get_optimizer(params,  all_optims[optim_name] + f",lr={lr}")

    if task == "regression" : criterion = nn.MSELoss()
    else : 
        if "multi" in task : criterion = nn.CrossEntropyLoss() 
        elif "bin" in task : criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = {}, {}
    train_accuracies, val_accuracies = {}, {}
    Rs, Qs, Ts, e_g = {}, {}, {}, {}
    T = (1/N)*(w_start.data @ F_matrix.T @ F_matrix @ w_start.data.T) # (M, N) x (N, N) x (N, N) x (N, M)
    T = T.cpu().numpy()
    try :
        for ep in (pbar := tqdm.tqdm(range(1, max_epoch+1))):
            train_loss, val_loss = [], []
            train_acc, val_acc = [], []
            ##########################################
            ##########################################
            # P = train_size
            # if weights_x is None : x = iid_normal(dim=N, sample_shape=(P,), mu=mu_x, sigma=sigma_x) # (P, N)
            # else : x = iid_mixture_normal(dim=N, sample_shape=(P,), mu=mu_x, sigma=sigma_x, weights=weights_x) # (P, N)
            # noise = iid_normal(dim=out_dim, sample_shape=(P,), mu=0.0, sigma=0.1) # (P, out_dim)
            # F_matrix = torch.eye(N)
            # if k is not None :
            #     F_matrix, _, _, _ = get_modulation_matrix_multi_singular(N, k, singular_val) # (N, N)
            #     F_matrix = torch.from_numpy(F_matrix).float()
            # x = x @ F_matrix # (P, N)
            # x = x.to(device)
            # y, _, _ = forward(w_start, v_start, x, N, g=act_funct) # (P, out_dim)
            # if "bin" in task : y = binarize(y)
            # logits, _, _ = forward(w, v, x, N, g=act_funct, denom=denom)
            # loss = criterion(input=logits, target=y)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # ##
            # epsilon_w = noise_optim*iid_normal(dim=N, sample_shape=(K,), mu=0.0, sigma=inv_tmp) # (K, N)
            # w.data = w.data + epsilon_w.to(device)
            # epsilon_v = noise_optim*iid_normal(dim=out_dim, sample_shape=(K,), mu=0.0, sigma=inv_tmp) # (K, out_dim)
            # v.data = v.data + epsilon_v.to(device)
            # ##
            # train_loss.append(loss.item())
            # if "classification" in task : train_acc.append(get_acc(logits, y, task))
            # val_loss.append(loss.item())
            # if "classification" in task : val_acc.append(get_acc(logits, y, task))
            # # with torch.no_grad():
            # #     P = val_size
            # #     if weights_x is None : x = iid_normal(dim=N, sample_shape=(P,), mu=mu_x, sigma=sigma_x) # (P, N)
            # #     else : x = iid_mixture_normal(dim=N, sample_shape=(P,), mu=mu_x, sigma=sigma_x, weights=weights_x) # (P, N)
            # #     noise = iid_normal(dim=out_dim, sample_shape=(P,), mu=0.0, sigma=0.1) # (P, out_dim)
            # #     x = x.to(device)
            # #     y, _, _ = forward(w_start, v_start, x, N, g=act_funct) # (P, out_dim)
            # #     logits, _, _ = forward(w, v, x, N, g=act_funct, denom=denom) 
            # #     loss = criterion(input = logits, target=y)
            # #     val_loss.append(loss.item())
            # #     if "classification" in task : val_acc.append(get_acc(logits, y, task))
            ##########################################
            ##########################################

            for x, y in train_loader :
                x, y = x.to(device), y.to(device)
                logits, _, _ = forward(w, v, x, N, g=act_funct, denom=denom)
                loss = criterion(input=logits, target=y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ##
                epsilon_w = noise_optim*iid_normal(dim=N, sample_shape=(K,), mu=0.0, sigma=inv_tmp) # (K, N)
                w.data = w.data + epsilon_w.to(device)
                epsilon_v = noise_optim*iid_normal(dim=out_dim, sample_shape=(K,), mu=0.0, sigma=inv_tmp) # (K, out_dim)
                v.data = v.data + epsilon_v.to(device)
                ##
                train_loss.append(loss.item())
                if "classification" in task : train_acc.append(get_acc(logits, y, task))
            train_losses[ep] = sum(train_loss)/len(train_loss)
            with torch.no_grad():
                for x, y in val_loader :
                    x, y = x.to(device), y.to(device)
                    logits, _, _ = forward(w, v, x, N, g=act_funct, denom=denom) 
                    loss = criterion(input = logits, target=y)
                    val_loss.append(loss.item())
                    if "classification" in task :
                        val_acc.append(get_acc(logits, y, task))
            ##########################################
            ##########################################

            train_losses[ep] = sum(train_loss)/len(train_loss)
            val_losses[ep] = sum(val_loss)/len(val_loss)
            pbar_infos = f"train_loss : {round(train_losses[ep],5)}, val_loss : {round(val_losses[ep],5)}"

            if "classification" in task :
                train_accuracies[ep] = sum(train_acc)/len(train_acc)
                val_accuracies[ep] = sum(val_acc)/len(val_acc)
                pbar_infos += f", train_acc : {round(train_accuracies[ep],5)}, val_acc : {round(val_accuracies[ep],5)}"
            pbar.set_description(pbar_infos)
            R = (1/N)*(w_start @ F_matrix @ w.data.T) # (M, N) x (N, N) x (N, K) = (M, K)
            Q = (1/N)*(w.data @ F_matrix.T @ F_matrix @ w.data.T) # (K, N) x (N, N) x (N, N) x (N, K)
            Rs[ep] = R.cpu().numpy()
            Qs[ep] = Q.cpu().numpy()
            Ts[ep] = T
            # e_g[ep] = (1/2)*(1+Q-2*R)

            if val_losses and val_losses[ep] < 0.00001 : break
    except KeyboardInterrupt:
        pass

    min_ep, max_ep = 0, max_epoch*2
    L, C = (2 if "classification" in task else 1), 2
    if Qs : L += 1
    #figsize=(C*15, L*10)
    figsize=(C*6, L*4)
    fig = plt.figure(figsize=figsize)
    for i, train_m, val_m, label, threshold in zip(
        [0] + ([2] if "classification" in task else []),
        [train_losses, train_accuracies],
        [val_losses, val_accuracies],
        ['loss', "accuracy"],
        [0.0, 1.0]
    ):
        for j in [1, 2] :
            ax = fig.add_subplot(L, C, i+j)
            x_, y_  = list(train_m.keys()), list(train_m.values())
            #x_ = [i/N for i in x_]
            ax.plot(x_[min_ep:max_ep], y_[min_ep:max_ep], label="train")
            if val_m :
                x_, y_  = list(val_m.keys()), list(val_m.values())
                #x_ = [i/N for i in x_]
                ax.plot(x_[min_ep:max_ep], y_[min_ep:max_ep],   label="val")
            ax.axhline(y = threshold, color = "red", linestyle = '--')
            ax.set_xlabel('epoch')
            ax.set_ylabel(label)
            ax.legend()
        ax.set_xscale("log")
        #ax.set_yscale("log")
        ax.set_xlabel('epoch (log scale)')

    if Qs :
        for k in [i+j+1, i+j+2] :
            ax = fig.add_subplot(L, C, k)
            x_, y_  = list(Rs.keys()), [el[0][0] for el in Rs.values()]
            ax.plot(x_[min_ep:max_ep], y_[min_ep:max_ep], label="Rs")
            x_, y_  = list(Qs.keys()), [el[0][0] for el in Qs.values()]
            ax.plot(x_[min_ep:max_ep], y_[min_ep:max_ep], label="Qs")
            x_, y_  = list(Ts.keys()), [el[0][0] for el in Ts.values()]
            ax.plot(x_[min_ep:max_ep], y_[min_ep:max_ep], label="Ts")
            #ax.plot(e_g, label="e_g")
            ax.set_xlabel('epoch')
            ax.set_ylabel("alignement")
            ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel('epoch (log scale)')

    # sauvegarder l'image
    save_path=f"D:/Canada/MILA/2023_path_to_PhD/double descent/images_scm"
    file_name = f'{N}_{g}_{train_size}_{val_size}_{task}_{optim_name}_{max_epoch}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{file_name}.png'), bbox_inches='tight')
    #plt.savefig(os.path.join(save_path, f'{file_name}.pdf'), bbox_inches='tight', format = "pdf")

    plt.show()