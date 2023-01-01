import torch
import pytorch_lightning as pl
import wandb
from loguru import logger

from src.dataset import LMLightningDataModule
from src.utils import bool_flag, to_none, str2dic_all, str2list, intorstr, str2list_func
from src.trainer import train

from argparse import ArgumentParser

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = ArgumentParser(description="Grokking for MLP")

    # Main parameters
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--exp_id", type=str, default=parser.parse_known_args()[0].task, help="Experiment id")
    parser.add_argument("--log_dir", type=str, help="Experiment dump path") # trainer

    # Model
    parser.add_argument("--c_out", type=str2list_func(int), help="out channels for CNN, eg 10,10") 
    parser.add_argument("--hidden_dim", type=str2list_func(int), help="hidden dim for FNN, eg 10") 
    parser.add_argument("--kernel_size", type=int, help="") 
    parser.add_argument("--kernel_size_maxPool", type=int, help="") 
    parser.add_argument("--dropout", type=float, default=0.0, help="")

    # Dataset
    parser.add_argument("--dataset_name", choices=["mnist", "fashion_mnist", "cifar10", "cifar100"])
    parser.add_argument("--train_batch_size", type=int, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, help="Validation batch size")
    parser.add_argument("--train_pct", type=int, default=100, help="training data fraction")
    parser.add_argument("--val_pct", type=int, default=100, help="val data fraction") 

    parser.add_argument("--limit_train_batches", type=float, default=1., help="limit batches for training data")
    parser.add_argument("--limit_val_batches", type=float, default=1., help="limit batches for validation data")
    parser.add_argument("--limit_test_batches", type=float, default=1., help="limit batches for test data")
    
    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adam,beta1=0.9,beta2=0.99,eps=0.00000001", help="""
                - optimizer parameters : adam_inverse_sqrt,beta1=0.9,beta2=0.99,eps=0.00000001 ...
                - classes : CustomAdam (custom_adam), Adam (adam), AdamInverseSqrtWithWarmup (adam_inverse_sqrt), AdamCosineWithWarmup (adam_cosine),
                            Adadelta (adadelta), Adagrad (adagrad), Adamax (adamax), ASGD (asgd), SGD (sgd), RMSprop (rmsprop), Rprop (rprop)
                - See https://pytorch.org/docs/stable/optim.html for optimizers parameters
    """) # optim : training
    parser.add_argument("--lr", type=float, default=1e-3, help="")
    # LR Scheduler
    parser.add_argument("--lr_scheduler", type=to_none, default="", help="""
                eg : 
                    - reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss
                    - constant_lr,factor=0.33,total_iters=5,last_epoch=-1
                Using a scheduler is optional but can be helpful.
                The scheduler reduces the LR if the validation performance hasn't improved for the last N (patience) epochs.
                class : reduce_lr_on_plateau, constant_lr, linear_lr, cosine_annealing_lr, exponential_lr, lambda_lr, 
                multiplicative_lr, step_lr, multi_step_lr, cyclic_lr, one_cycle_lr, cosine_annealing_warm_restarts            
    """)

    # Training
    parser.add_argument("--max_epochs", type=int, help="Maximun number of epoch")

    parser.add_argument("--validation_metrics", type=str, default="val_acc", help="Validation metrics : val_acc, val_loss, val_ppl ...") # trainer

    parser.add_argument("--checkpoint_path", type=to_none, default=None, help="Load pretrained model from checkpoint and continued training")
    parser.add_argument("--model_name", type=str, default="", 
                        help="Load a model from root directory (`log_dir/exp_id`), eg : epoch=88-val_loss=13.6392.ckpt")
    parser.add_argument("--every_n_epochs", type=int, 
                        help="Frequency at which the images of the representations and the last layer weights are sent to wandb")
    parser.add_argument("--early_stopping_patience", type=int, default=1e9, 
                        help="Early stopping patience : If the model does not converge during these numbers of steps, stop the training") 
    parser.add_argument("--save_top_k", type=int, default=-1, 
                        help="The best `save_top_k` models according to the quantity monitored will be saved.\
                              If save_top_k == 0, no models are saved. if save_top_k == -1, all models are saved.")

    # Wandb
    parser.add_argument("--use_wandb", type=bool_flag, default=False, help="") 
    parser.add_argument("--wandb_entity", type=str, help="") 
    parser.add_argument("--wandb_project", type=str, help="")
    parser.add_argument("--group_name", type=to_none, default=None, help="")
    parser.add_argument("--group_vars", type=str2list, help="") 

    # Devices & Seed
    parser.add_argument("--accelerator", type=str, default="auto", help="accelerator types : cpu, gpu, tpu, ipu, auto") 
    parser.add_argument("--devices", type=intorstr, default="auto", help="number of cpu processes, of gpu/tpu cores ...")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproductibility")

    # Early_stopping (stop training after grokking)
    parser.add_argument("--early_stopping_grokking", type=str2dic_all, default="", help="""
        * eg. : "patience=int(1000),metric=str(val_acc),metric_threshold=float(90.0)"
        * Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold`"
        """)

    return parser

def main(params) :

    pl.seed_everything(params.random_seed, workers=True)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Dataset
    params.regression = params.task == "regression"

    logger.info("Data module")
    data_module = LMLightningDataModule(
        dataset_name = params.dataset_name,
        train_batch_size = params.train_batch_size,
        val_batch_size = params.val_batch_size,
        train_pct = params.train_pct,
        val_pct = params.val_pct,
        #data_path = params.log_dir + "/data"
        #num_workers = params.num_workers,
    )

    setattr(params, "data_infos", data_module.data_infos)
    setattr(params, "train_dataset", data_module.train_dataset)

    torch.save(data_module, params.log_dir + "/data.pt")
    torch.save(params, params.log_dir + "/params.pt")

    # Train
    logger.info("Model")
    model, result = train(params, data_module)
    
    print("\n********")
    print(result)
    print("********\n")

if __name__ == "__main__":
    # generate parser / parse parameters
    params = get_parser().parse_args()
    print()
    for k, v in vars(params).items() : print(k, " --> ", v)
    print()

    # run experiment
    main(params)
