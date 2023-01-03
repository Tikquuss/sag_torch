import itertools
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pytorch_lightning as pl
import torch
import os
from src.utils import AttrDict, GROUP_VARS
from src.dataset import LMLightningDataModule
from src.utils import get_group_name
from src.trainer import train

def plot_results(params, model_dict, hparms_1, hparms_2, s1, s2):
    """
    2D plot of train&val acc&loss as a function of two parameters use for phase diagram
    """
    fig = plt.figure()
    fig.suptitle("Grokking")

    figsize=(2*8, 6)
    plt.gcf().set_size_inches(figsize)

    i = 1
    for metric in (["loss"] if params.data_infos["task"] == "regression" else ["loss", "acc"])  :
        ax = fig.add_subplot(1, 2, i, projection='3d')
        i += 1 
        xs, ys, zs = [], [], []
        for split, (m, zlow, zhigh) in zip(["val", "train"], [('o', -50, -25), ('^', -30, -5)]) :
            for a, b in itertools.product(hparms_1, hparms_2) :
                k = f"{s1}={a}, {s2}={b}"
                if k in model_dict.keys():
                    xs.append(a)
                    ys.append(b)
                    #print(k, f"{split}_{metric}", model_dict[k]["result"][split][f"{split}_{metric}"])
                    zs.append(model_dict[k]["result"][split][f"{split}_{metric}"])

            ax.scatter(xs, ys, zs, marker=m, label = split)

        ax.set_xlabel(s1)
        ax.set_ylabel(s2)
        ax.set_zlabel(metric)
        ax.set_title(metric, fontsize=14)
        ax.legend()
    plt.show()

if __name__ == "__main__":
    
    weight_decay=0.0
    lr=0.001
    dropout=0.5
    opt="adam"
    group_name=f"wd={weight_decay}-lr={lr}-d={dropout}-opt={opt}"

    random_seed=0
    log_dir="../log_files"

    dataset_name="iris"
    train_pct=80

    #val_metric="val_acc"
    val_metric="val_loss"

    opt=f"{opt},weight_decay={weight_decay},beta1=0.9,beta2=0.99,eps=0.00000001"
    opt="sag"
    opt=f"sgd,weight_decay={weight_decay}"
    opt=f"sag,weight_decay={weight_decay},batch_mode=False,init_y_i=True"


    params = AttrDict({
        ### Main parameters
        "exp_id" : f"{dataset_name}",
        "log_dir" : f"{log_dir}",

        ### Model
        "c_out" :  [10, 10],
        "hidden_dim" :  [50],
        "kernel_size" : [5],
        "kernel_size_maxPool" : 2,
        "dropout"  : dropout,

        ### Dataset
        "dataset_name":dataset_name,
        "train_batch_size" : 512,
        "val_batch_size" : 512,
        "train_pct" : train_pct,
        "val_pct" : 100,

        ### Optimizer
        "optimizer" : opt,
        "lr" : lr,

        ### LR Scheduler
        "lr_scheduler" : None,
        #"lr_scheduler" : "reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss",
        
        ### Training
        "max_epochs" : 10, 
        "validation_metrics" : "val_loss",
        "checkpoint_path" : None, 
        "model_name": "", 
        "every_n_epochs":1, 
        "every_n_epochs_show":1, 
        "early_stopping_patience":1e9, 
        "save_top_k":-1,

        # Wandb 
        "use_wandb" : False,
        "wandb_entity" : "grokking_ppsp",
        "wandb_project" : f"dataset={dataset_name}",
        "group_name" : group_name,

        "group_vars" : None,
        
        # Devices & Seed
        "accelerator" : "auto",
        "devices" : "auto",
        "random_seed": random_seed,

        ### Early_stopping (for grokking) : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold` 
        #"early_stopping_grokking" : None,
        "early_stopping_grokking" : f"patience=int(1000),metric=str({val_metric}),metric_threshold=float(90.0)",

    })

    pl.seed_everything(params.random_seed, workers=True)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    root_dir = os.path.join(params.log_dir, params.exp_id, params.group_name, str(params.random_seed)) 
    os.makedirs(root_dir, exist_ok=True)

    data_module = LMLightningDataModule(
        dataset_name = params.dataset_name,
        train_batch_size = params.train_batch_size,
        val_batch_size = params.val_batch_size,
        train_pct = params.train_pct,
        val_pct = params.val_pct,
        data_path = params.log_dir + "/data",
        #num_workers = params.num_workers,
    )
    setattr(params, "data_infos", data_module.data_infos)
    setattr(params, "train_dataset", data_module.train_dataset)

    ######## Example : phase diagram with representation_lr and decoder_lr/weight_decay

    lrs = [1e-2, 1e-3, 1e-4, 1e-5] 
    #lrs = np.linspace(start=1e-1, stop=1e-5, num=10)

    weight_decays = [0, 1]
    #weight_decays = list(range(20))
    #weight_decays =  np.linspace(start=0, stop=20, num=21)

    s = "weight_decay"
    assert s in params["optimizer"]
    print(lrs, weight_decays)

    model_dict = {}
    i = 0
    for a, b in itertools.product(lrs, weight_decays) :

        params["lr"] = a 
        params["optimizer"] = params["optimizer"].replace(f"{s}={weight_decay}", f"{s}={b}")
    
        name = f"lr={a}, {s}={b}"
        params.exp_id = name
        
        #group_vars = GROUP_VARS + ["lr", s]
        group_vars = ["lr", s]
        group_vars = list(set(group_vars))
        setattr(params, s, b)
        params["group_name"] = get_group_name(params, group_vars = group_vars)
        
        print("*"*10, i, name, "*"*10)
        i+=1

        model, result = train(params, data_module, root_dir)
        
        model_dict[name] = {"model": model, "result": result}

    ########

    print(model_dict.keys())

    val_loss = [model_dict[k]["result"]["val"]["val_loss"] for k in model_dict]
    val_acc = [model_dict[k]["result"]["val"].get("val_acc", 0) for k in model_dict]
    print(val_loss, val_acc)

    plot_results(params, model_dict, 
        hparms_1 = lrs, hparms_2 = weight_decays,
        s1 = 'lr', s2 = s
    )

    ########

    # for k in model_dict :
    #     print("*"*10, k, "*"*10)
    #     model = model_dict[k]["model"]
    #     # TODO