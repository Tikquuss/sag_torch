import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os

from src.optim import get_optimizer
from src.datasets.msf import get_dataloader, get_weights, forward, iid_normal, get_data
from src.dataset import g_dic
from src.optim import get_all_optims


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    seed=100
    np.random.seed(seed)
    torch.manual_seed(seed)

    N=500
    alpha = .5
    s={"N":N, "out_dim":1, "g":"sigmoid", "noise":0.001}
    train_size, val_size = int(alpha*N), 1000

    k = [10, 50]
    #k = {1.5 : 0.5}
    #k = {5 : 0.2, 10 : 0.2, 50:0.2, 100:0.2} # 1000:0.2}
    #k = None
    singular_val=1.0

    max_epoch = 10000*50#00
    noise_optim, inv_tmp = 0.0, 2/100000#00
    lr = 0.1
    weight_decay=0.0

    task = "regression"
    #task = "classification"

    train_batch_size, val_batch_size = 2**20, 2**20
    mu_w, sigma_w = 0.0, 1.0 # w
    N, g = s["N"], s["g"]
    noise, out_dim = s.get("noise", 0.0), s.get("out_dim", 1)
    act_funct = g_dic[g]
    
    #if g in ["relu", 'id'] : sigma_w = N**-0.5
    sigma_w = 1
    
    denom = 1.0
    sigma_x = 1/np.sqrt(N)
    
    # denom = np.sqrt(N)
    # sigma_x = 1/np.sqrt(N)
    
    _, _, data_infos, w_start, F_matrix, x_train, y_train, x_test, y_test = get_dataloader(
        train_size, val_size, N,
        k=k, singular_val=singular_val,
        out_dim = out_dim, g=act_funct,
        mu_x = 0.0, sigma_x = sigma_x, # data
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
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    for key, val in data_infos.items()  : print(str(key) + " --> " + str(val))

    w = get_weights(N, out_dim = out_dim, mu_w = mu_w, sigma_w = sigma_w)
    #w.requires_grad_(True)
    w=w.to(device)
    #w=torch.nn.Parameter(w)

    if task == "regression" : criterion = nn.MSELoss()
    else :
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    Rs, Qs, Ts, e_g = [], [], [], []
    T = (1/N)*(w_start.T @ F_matrix.T @ F_matrix @ w_start.data) # (out_dim, N) x (N, N) x (N, N) x (N, out_dim)

    XF = x_train @ F_matrix # P x N
    const = (1/train_size)
    A = const * (XF.T @ XF) + weight_decay # NxN
    B = const * (XF.T @ x_train) @ w_start # Nx1
    
    H = torch.eye(N).to(device) - lr * A
    #e_values, e_vect = torch.linalg.eig(H)
    e_values, e_vect = torch.linalg.eigh(H) # for Hermitian and symmetric matrices.
    #print(e_values)

    def f_map(w) : return w - lr * (A @ w - B)
    def grad_f_map(w) : return H

    try :
        w_gd = torch.linalg.inv(A) @ B
        #print(w_gd)
        #print(w)
        with torch.no_grad():
            y_hat, _ = forward(w_gd, x_train, N, g=act_funct, denom=denom)
            loss = F.mse_loss(input=y_hat, target=y_train, size_average=None, reduce=None, reduction='mean')
            print(loss)
            y_hat, _ = forward(w_gd, x_test, N, g=act_funct, denom=denom)
            loss = F.mse_loss(input=y_hat, target=y_test, size_average=None, reduce=None, reduction='mean')
            print(loss)
    except :
        print("================= B^-1 impossible")
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    Rs, Qs, Ts, e_g = [], [], [], []
    T = (1/N)*(w_start.T @ F_matrix.T @ F_matrix @ w_start.data) # (out_dim, N) x (N, N) x (N, N) x (N, out_dim)
    try :
        for ep in (pbar := tqdm.tqdm(range(max_epoch))):
            with torch.no_grad():
                y_hat, _ = forward(w, x_train, N, g=act_funct, denom=denom)
                loss = F.mse_loss(input=y_hat, target=y_train, size_average=None, reduce=None, reduction='mean')
                
                epsilon = noise_optim*iid_normal(dim=out_dim, sample_shape=(N,), mu=0.0, sigma=inv_tmp)
                w = f_map(w) + epsilon.to(device)
                train_losses.append(loss.item())

                y_hat, _ = forward(w, x_test, N, g=act_funct, denom=denom)
                loss = F.mse_loss(input=y_hat, target=y_test, size_average=None, reduce=None, reduction='mean')
                val_losses.append(loss.item())
            
            pbar_infos = f"train_loss : {round(train_losses[-1],5)}, val_loss : {round(val_losses[-1],5)}"
            pbar.set_description(pbar_infos)

            R = (1/N)*(w_start.T @ F_matrix @ w.data) # (out_dim, N) x (N, N) x (N, out_dim)
            Q = (1/N)*(w.T @ F_matrix.T @ F_matrix @ w.data) # (out_dim, N) x (N, N) x (N, N) x (N, out_dim)
            Rs.append(R.item())
            Qs.append(Q.item())
            Ts.append(T.item())
            e_g.append((1/2)*(1+Q.item()-2*R.item()))

            if val_losses[-1] < 0.0001 : break
                                        
    except KeyboardInterrupt:
        pass

    #print(w)

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
    save_path=f"D:/Canada/MILA/2023_path_to_PhD/double descent/images_msf2"
    file_name = f'{N}_{g}_{train_size}_{val_size}_{task}_{max_epoch}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{file_name}.png'), bbox_inches='tight')
    #plt.savefig(os.path.join(save_path, f'{file_name}.pdf'), bbox_inches='tight', format = "pdf")

    plt.show()