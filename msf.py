import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os

from src.optim import get_optimizer
from src.datasets.msf import get_dataloader, get_weights, forward, iid_normal
from src.dataset import g_dic
from src.optim import get_all_optims

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    seed=100
    np.random.seed(seed)
    torch.manual_seed(seed)

    s={"N":1000, "out_dim":1, "g":"id", "noise":0.0001}
    train_size, val_size = 999+2, 1000

    k = [10, 50]
    k = {1000 : 0.2}
    #k = {5 : 0.2, 10 : 0.2, 50:0.2, 100:0.2, 1000:0.2}
    k = None
    singular_val=1.0

    max_epoch = 1000*5#00
    noise_optim, inv_tmp = 1.0, 2/100000#00
    lr = 0.01
    optim_name="custom_adam"
    #optim_name="sgd"
    all_optims = get_all_optims(weight_decay=0.0, momentum=0.9, beta1=0.9, beta2=0.99)

    task = "regression"
    #task = "classification"

    train_batch_size, val_batch_size = 2**20, 2**20
    mu_w, sigma_w = 0.0, 1.0 # w
    N, g = s["N"], s["g"]
    noise, out_dim = s.get("noise", 0.0), s.get("out_dim", 1)
    act_funct = g_dic[g]
    if g in ["relu", 'id'] : sigma_w = N**-0.5
    denom = 1.0
    #denom = np.sqrt(N)
    train_loader, val_loader, data_infos, w_start, F_matrix = get_dataloader(
        train_size, val_size, N,
        k=k, singular_val=singular_val,
        out_dim = out_dim, g=act_funct,
        mu_x = 0.0, sigma_x = 1.0, # data
        mu_w = mu_w, sigma_w = sigma_w, # w
        mu_noise = 0.0, sigma_noise = noise, # noise
        seed = seed, task = task,
        include_indexes = False,
        train_batch_size = train_batch_size, val_batch_size = val_batch_size,
        num_workers=0,
        return_just_set = False
    )
    F_matrix = F_matrix.to(device)
    w_start = w_start.to(device)
    for key, val in data_infos.items()  : print(str(key) + " --> " + str(val))

    w = get_weights(N, out_dim = out_dim, mu_w = mu_w, sigma_w = sigma_w)
    w.requires_grad_(True)
    w=w.to(device)
    w=torch.nn.Parameter(w)

    params = [{'params':[w], 'lr':lr}]
    optimizer = get_optimizer(params,  all_optims[optim_name] + f",lr={lr}")

    if task == "regression" :
        criterion = nn.MSELoss()
    else :
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    Rs, Qs, Ts, e_g = [], [], [], []
    T = (1/N)*(w_start.T @ F_matrix.T @ F_matrix @ w_start.data) # (out_dim, N) x (N, N) x (N, N) x (N, out_dim)
    try :
        for ep in (pbar := tqdm.tqdm(range(max_epoch))):
            train_loss, val_loss = [], []
            train_acc, val_acc = [], []
            for x, y in train_loader :
                x, y = x.to(device), y.to(device)
                y_hat, _ = forward(w, x, N, g=act_funct, denom=denom)
                loss = F.mse_loss(input=y_hat, target=y, size_average=None, reduce=None, reduction='mean')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epsilon = noise_optim*iid_normal(dim=out_dim, sample_shape=(N,), mu=0.0, sigma=inv_tmp)
                w.data = w.data + epsilon.to(device)
                train_loss.append(loss.item())
                if "classification" in task :
                    acc = (F.sigmoid(y_hat).round() == y).float().mean().item()
                    train_acc.append(acc)

            with torch.no_grad():
                for x, y in val_loader :
                    x, y = x.to(device), y.to(device)
                    y_hat, _ = forward(w, x, N, g=act_funct, denom=denom)
                    loss = F.mse_loss(input=y_hat, target=y, size_average=None, reduce=None, reduction='mean')
                    val_loss.append(loss.item())
                    if "classification" in task :
                        acc = (F.sigmoid(y_hat).round() == y).float().mean().item()
                        val_acc.append(acc)

            train_losses.append(sum(train_loss)/len(train_loss))
            val_losses.append(sum(val_loss)/len(val_loss))
            pbar_infos = f"train_loss : {round(train_losses[-1],5)}, val_loss : {round(val_losses[-1],5)}"
            if "classification" in task :
                train_accuracies.append(sum(train_acc)/len(train_acc))
                val_accuracies.append(sum(val_acc)/len(val_acc))
                pbar_infos += f", train_acc : {round(train_accuracies[-1],5)}, val_acc : {round(val_accuracies[-1],5)}"
            pbar.set_description(pbar_infos)

            R = (1/N)*(w_start.T @ F_matrix @ w.data) # (out_dim, N) x (N, N) x (N, out_dim)
            Q = (1/N)*(w.T @ F_matrix.T @ F_matrix @ w.data) # (out_dim, N) x (N, N) x (N, N) x (N, out_dim)
            Rs.append(R.item())
            Qs.append(Q.item())
            Ts.append(T.item())
            e_g.append((1/2)*(1+Q.item()-2*R.item()))

            if val_losses[-1] < 0.001 : break
            
    except KeyboardInterrupt:
        pass

    L, C = (2 if "classification" in task else 1)+1, 2
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
            ax.plot(train_m, label="train")
            ax.plot(val_m,   label="val")
            ax.axhline(y = threshold, color = "red", linestyle = '--')
            ax.set_xlabel('epoch')
            ax.set_ylabel(label)
            ax.legend()
        ax.set_xscale("log")
        ax.set_xlabel('epoch (log scale)')

    for k in [i+j+1, i+j+2] :
        ax = fig.add_subplot(L, C, k)
        ax.plot(Rs, label="Rs")
        ax.plot(Qs, label="Qs")
        ax.plot(Ts, label="Ts")
        #ax.plot(e_g, label="e_g")
        ax.set_xlabel('epoch')
        ax.set_ylabel("alignment")
        ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel('epoch (log scale)')

    # sauvegarder l'image
    save_path=f"D:/Canada/MILA/2023_path_to_PhD/double descent/images_msf"
    file_name = f'{N}_{g}_{train_size}_{val_size}_{task}_{optim_name}_{max_epoch}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{file_name}.png'), bbox_inches='tight')
    #plt.savefig(os.path.join(save_path, f'{file_name}.pdf'), bbox_inches='tight', format = "pdf")

    plt.show()