# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# PyTorch Lightning
import pytorch_lightning as pl
# wandb
import wandb
import itertools
import math
import os
from typing import List, Union, Dict
from loguru import logger

from .optim import configure_optimizers
from .dataset import SKLEAN_SET, TORCH_SET
from .hash import get_hash_path

possible_metrics = ["%s_%s"%(i, j) for i, j in itertools.product(["train", "val"], ["acc", "loss"])]

# MLP
def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

class MLP4Regression(nn.Module):
    def __init__(self, l, act=nn.LeakyReLU(), tail=[], dropout=0.0):
        super(MLP4Regression, self).__init__()
        self.net = make_mlp(l, act, tail = tail + [nn.Dropout(dropout)])
    def forward(self, x):
        """
        x: (bs, _)
        """
        return self.net(x) # (bs, _)

class MLP4Classif(nn.Module):
    def __init__(self, l, act=nn.LeakyReLU(), tail=[], dropout=0.0):
        super(MLP4Classif, self).__init__()
        self.net = make_mlp(l, act, tail = tail + [nn.Dropout(dropout)])
    def forward(self, x):
        """
        x: (bs, _)
        """
        x = self.net(x) # (bs, n_class)
        return F.softmax(x, dim=-1)
        return F.log_softmax(x, dim=-1)

# Text
def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

# Images
class CNNNet(nn.Module):
    def __init__(self, c_in, h_in, w_in, c_out : List, kernel_size, hidden_dim : List, n_class, kernel_size_maxPool=2, dropout=0.0):
        super(CNNNet, self).__init__()

        assert len(c_out) >= 1
        assert len(hidden_dim) >= 1

        stride=1
        padding=0
        dilation=1
        stride_maxPool=kernel_size_maxPool

        def get_out_dim_conv(h_in, w_in, kernel_size, stride, padding, dilation):
            h_out = math.floor((h_in + 2 * padding - dilation*(kernel_size - 1) - 1) / stride + 1)
            w_out = math.floor((w_in + 2 * padding - dilation*(kernel_size - 1) - 1) / stride + 1)
            return h_out, w_out

        self.conv = []
        self.conv.append(
            nn.Conv2d(
                in_channels=c_in, 
                out_channels=c_out[0],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )   
        )
        h_out, w_out = get_out_dim_conv(h_in, w_in, kernel_size, stride, padding, dilation, )
        self.conv.append(
            nn.MaxPool2d(kernel_size_maxPool, stride_maxPool, padding, dilation) 
        )
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size_maxPool, stride_maxPool, padding, dilation)
        for i in range(1, len(c_out)) :
            self.conv.append(
                nn.Conv2d(
                    in_channels=c_out[i-1], 
                    out_channels=c_out[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation
                )   
            )
            h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size, stride, padding, dilation)
            self.conv.append(nn.Dropout2d(p=dropout))
            self.conv.append(
                nn.MaxPool2d(kernel_size_maxPool, stride_maxPool, padding, dilation) 
            )
            h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size_maxPool, stride_maxPool, padding, dilation)

        self.conv = nn.Sequential(*self.conv)

        self.fc_input_dim = h_out*w_out*c_out[-1]
        self.fc = []
        hidden_dim = hidden_dim + [n_class]
        self.fc.append(nn.Linear(self.fc_input_dim, hidden_dim[0]))
        for i in range(1, len(hidden_dim)) :
            self.fc.append(nn.Dropout(p=dropout))
            self.fc.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
        self.fc.append(nn.Dropout(p=dropout))
        self.fc = nn.Sequential(*self.fc)
        
    def forward(self, x):
        """
        x: (bs, c_in, h_in, w_in) or (c_in, h_in, w_in)
        """
        x = self.conv(x) # (bs, c_out, h_out, w_out)
        x = x.view(-1, self.fc_input_dim) # (bs, c_out*h_out*w_out)
        x = self.fc(x) # (bs, n_class)
        return F.log_softmax(x, dim=-1)

class Model(pl.LightningModule):
    """
    params : 
        - hidden_dim (int), regression (bool)
        - use_wandb (int, optional, False)
        - lr (float, optional, 1e-3), weight_decay (float, optional, 0) 
        - patience (float, optional, 20), min_lr (float, optional, 5e-5)
    """
    def __init__(self, params):
        """
        Transformer model 
        """
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters(params) 
        # model and loss
        regression = self.hparams.data_infos["task"] == "regression"
        if self.hparams.dataset_name in TORCH_SET :
            self.backbone = CNNNet(
                self.hparams.data_infos["c_in"], 
                self.hparams.data_infos["h_in"], 
                self.hparams.data_infos["w_in"], 
                self.hparams.c_out, 
                self.hparams.kernel_size, 
                self.hparams.hidden_dim, 
                self.hparams.data_infos["n_class"], 
                self.hparams.kernel_size_maxPool, 
                self.hparams.dropout
            )
        if self.hparams.dataset_name in SKLEAN_SET :
            _class = MLP4Regression if regression else MLP4Classif
            self.backbone = _class(
                l = [self.hparams.data_infos["c_in"]] +  self.hparams.hidden_dim + [self.hparams.data_infos["n_class"]],
                act=nn.LeakyReLU(), 
                tail=[],
                dropout = self.hparams.dropout
            )
            
        if not regression :
            self.criterion = torch.nn.NLLLoss() if self.hparams.dataset_name in TORCH_SET else nn.CrossEntropyLoss() 
        else : 
            self.criterion = nn.MSELoss()

        self.use_wandb = self.hparams.use_wandb

        # State
        self.grok = False
        self.comprehension = False
        self.memorization = False
        self.confusion = True
        self.comp_epoch = float("inf")
        self.memo_epoch = float("inf")

        # Early stopping : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold`
        early_stopping_grokking = self.hparams.early_stopping_grokking
        if type(early_stopping_grokking) != dict : early_stopping_grokking = {} 
        self.es_patience = early_stopping_grokking.get("patience", self.hparams.max_epochs)
        self.es_metric = early_stopping_grokking.get("metric", "val_loss" if self.hparams.data_infos["task"] == "regression" else "val_acc") 
        assert self.es_metric in possible_metrics
        self.es_metric_threshold = early_stopping_grokking.get("metric_threshold", 0.0 if 'loss' in self.es_metric else 99.0) 
        self.es_mode = (lambda s : "min" if 'loss' in s else 'max')(self.es_metric)
        self.es_step = 0
        self.reached_limit = False

    def configure_optimizers(self):
        logger.info("Configure Optimizers and LR Scheduler")
        lr = self.hparams.get("lr", 1e-3)
        self.hparams.optimizer += f",lr={lr}"
        parameters = [{'params': self.backbone.parameters(), 'lr': lr}]
        if 'sag' in self.hparams.optimizer:
            n = self.hparams.data_infos['train_size']
            m = self.hparams.data_infos['train_n_batchs'] 
            self.hparams.optimizer += f",n={n},m={m}"
        optim_scheduler = configure_optimizers(parameters, self.hparams.optimizer, self.hparams.lr_scheduler)
        if 'sag' in self.hparams.optimizer and optim_scheduler["optimizer"].init_y_i :
            optim_scheduler["optimizer"] = self.init_y_i(parameters, optim_scheduler["optimizer"])
        return optim_scheduler

    def init_y_i(self, parameters, optimizer):
        logger.info("init_y_i")
        f = '%s_%s_%s_%s'%(
            self.hparams.optimizer, self.hparams.train_batch_size, self.hparams.val_batch_size, self.hparams.train_pct
        )
        opt_path = get_hash_path(self.hparams, f, prefix="optim", suffix="")
        #
        if os.path.isfile(opt_path) :
            optimizer2 = configure_optimizers(parameters, self.hparams.optimizer, self.hparams.lr_scheduler)["optimizer"]
            optimizer2.load_state_dict(torch.load(opt_path))
            for group, group2 in zip(optimizer.param_groups, optimizer2.param_groups):
                    for p, p2 in zip(group['params'], group2['params']):
                        if p.grad is None: continue
                        optimizer.state[p]['y_i'] = optimizer2.state[p2]['y_i']
            return optimizer
        #
        device = self.device
        batch_mode = optimizer.batch_mode
        batch_size = self.hparams.train_batch_size if batch_mode else 1
        train_loader = torch.utils.data.DataLoader(self.hparams.train_dataset, batch_size=batch_size, shuffle=False)
        #  delta_f (y_i) & delta_g
        """
        Since $\nabla g = \frac{1}{n} \sum_{i=1}^n \nabla f_i$ with potentially high $n$, 
        it would be impractical to calculate all $ \nabla f_i$'s before averaging. We proceed iteratively as follows.
        If we want to calculate $x = \frac{1}{n} \sum_{i=1}^n x_i$ for a given sequence $\{x_1, \dots, x_n\}$, we can 
        set $a_k = \frac{1}{k} \sum_{i=1}^n x_i$ and notice that $a_k = \frac{k-1}{k} a_{k-1} + \frac{1}{k} x_k$. 
        This allows us to compute $a_k$ iteratively up to rank $n$ and to set $x = a_n$.
        More precisely, 
            - $x = 0$, 
            - and for $k$ in $\{1, ...., n\}$ : 
                - $x = \frac{k-1}{k} x + \frac{1}{k} x_k$.
        """
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                state['delta_g'] = torch.zeros_like(p.data, device = p.data.device)
        #
        k = 0
        for batch_idx, (data, target, indexes) in enumerate(train_loader):
            loss, _, _, _ = self._get_loss(batch=(data.to(device), target.to(device), indexes))
            self.zero_grad()
            loss.backward()
            k+=1
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    grad = p.grad.data
                    state = optimizer.state[p]
                    state['y_i'][batch_idx if batch_mode else indexes] = grad + 0.0
                    state['delta_g'].add_(state['delta_g'], alpha = (k-1.0)/k).add_(grad, alpha = 1.0/k)
        #
        for batch_idx, (data, target, indexes) in enumerate(train_loader):
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    state = optimizer.state[p]
                    state['y_i'][batch_idx if batch_mode else indexes].sub_(state['delta_g']) 
        #
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                #optimizer.state[p].pop('delta_g', None)
                del optimizer.state[p]['delta_g']
        #
        torch.save(optimizer.state_dict(), opt_path)
        
        return optimizer

    def forward(self, x):
        """
        Inputs: `x`, Tensor(bs, 1, 28, 28), 
        """
        return self.backbone(x) # (bs, n_class)
    
    def _get_loss(self, batch):
        """
        Given a batch of data, this function returns the  loss (MSE or CEL)
        """
        x, y, indexes = batch # We do not need the labels
        tensor = self.forward(x)
        loss = self.criterion(input = tensor, target=y)
        return loss, tensor, y, indexes
    
    def training_step(self, batch, batch_idx):
        loss, tensor, y, indexes = self._get_loss(batch)

        if not self.automatic_optimization :
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            try :
                opt.step(batch_idx=batch_idx, indexes=indexes)
            except TypeError :
                opt.step()

            sch = self.lr_schedulers()
            if sch is not None :
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step(self.trainer.callback_metrics[sch.monitor])
                else:
                    sch.step()

        self.log('train_loss', loss, prog_bar=True)
        output = {"loss" : loss}
        if not (self.hparams.data_infos["task"] == "regression") : 
            acc = (tensor.argmax(dim=-1) == y).float().mean() * 100
            output["train_acc"] = acc
            self.log('train_acc', acc, prog_bar=True)
        return output 
    
    def validation_step(self, batch, batch_idx):
        loss, tensor, y, _ = self._get_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
        output = {'val_loss' : loss}
        if not (self.hparams.data_infos["task"] == "regression") : 
            acc = (tensor.argmax(dim=-1) == y).float().mean() * 100
            output["val_acc"] = acc
            self.log('val_acc', acc, prog_bar=True)
        return  output 
    
    def test_step(self, batch, batch_idx):
        loss, tensor, y, _ = self._get_loss(batch)
        #self.log('test_loss', loss, prog_bar=True)
        output = {'test_loss' : loss}
        if not (self.hparams.data_infos["task"] == "regression") : 
            acc = (tensor.argmax(dim=-1) == y).float().mean() * 100
            output["test_acc"] = acc
            #self.log('test_acc', acc, prog_bar=True)
        return output 

    def increase_es_limit(self, logs):
        es_metric = logs[self.es_metric]
        self.reached_limit = self.reached_limit or (es_metric >= self.es_metric_threshold if self.es_mode == "max" 
                                                    else es_metric <= self.es_metric_threshold)
        if self.reached_limit : self.es_step+=1
        return self.es_step

    def training_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"train_loss": loss}

        if 'train' in self.es_metric : logs["e_step"] = self.increase_es_limit(logs)

        if self.hparams.data_infos["task"] == "regression" : 
            memo_condition = round(loss.item(), 10) == 0.0
        else : 
            accuracy = torch.stack([x["train_acc"] for x in outputs]).mean()
            logs["train_acc"] = accuracy
            memo_condition = accuracy >= 99.0

        self.memorization = self.memorization or memo_condition
        if memo_condition : self.memo_epoch = min(self.current_epoch, self.memo_epoch)
        
        logs["train_epoch"]  = self.current_epoch

        schedulers = self.lr_schedulers()
        if schedulers is not None :
            try : scheduler = schedulers[0]
            except TypeError: scheduler = schedulers # 'xxx' object is not subscriptable
            param_groups = scheduler.optimizer.param_groups
            logs["representation_lr"] = param_groups[0]["lr"]
            logs["decoder_lr"] = param_groups[1]["lr"]

        for k, v in logs.items() : self.log(k, v, prog_bar=True)
        if self.use_wandb: wandb.log(logs)

    def validation_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        """
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {
            "val_loss": loss,    
            #"val_epoch": self.current_epoch,
        }

        if self.hparams.data_infos["task"] == "regression" : 
            comp_condition = round(loss.item(), 10) == 0.0
        else : 
            accuracy = torch.stack([x["val_acc"] for x in outputs]).mean()
            logs["val_acc"] = accuracy
            comp_condition = accuracy >= 99.0
 
        self.comprehension = self.comprehension or comp_condition
        if comp_condition : self.comp_epoch = min(self.current_epoch, self.comp_epoch)

        if 'val' in self.es_metric : logs["e_step"] = self.increase_es_limit(logs)

        for k, v in logs.items() : self.log(k, v, prog_bar=True)
        if self.use_wandb: wandb.log(logs)

        self.grok = self.comprehension and True # and long step of training
        self.memorization = (not self.comprehension) and self.memorization
        self.confusion = (not self.comprehension) and (not self.memorization)

        # diff_epoch = self.comp_epoch - self.memo_epoch
        # if not math.isnan(diff_epoch) : 
        #     self.grok = diff_epoch >= 100
        #     self.comprehension = not self.grok

        self.states = {
            "grok":self.grok, "comprehension":self.comprehension, "memorization": self.memorization, "confusion":self.confusion,
            "comprehension_epoch":self.comp_epoch, "memorization_epoch":self.memo_epoch
        }

    def send_dict_to_wandb(self, data, label, title) :
        if self.hparams.use_wandb:  
            labels = data.keys()
            values = data.values()
            data = [[label, val] for (label, val) in zip(labels, values)]
            table = wandb.Table(data=data, columns = ["label", "value"])
            wandb.log({label : wandb.plot.bar(table, "label", "value", title=title)})
    
    def on_train_start(self):
        db_data = getattr(self.hparams, "data_infos", None)
        if db_data is not None : self.send_dict_to_wandb(db_data, label = "data_info", title="Dataset Informations")

    def on_train_end(self) :

        # diff_epoch = self.comp_epoch - self.memo_epoch
        # if not math.isnan(diff_epoch) : 
        #     self.grok = diff_epoch >= 100
        #     self.comprehension = not self.grok

        states = {
            "grok":int(self.grok), "comprehension":int(self.comprehension), "memorization":int(self.memorization), "confusion":int(self.confusion),
            "comprehension_epoch":self.comp_epoch, "memorization_epoch":self.memo_epoch
        }
        self.send_dict_to_wandb(states, label = "states_info", title="Phase Informations")

if __name__ == "__main__":
    from .utils import AttrDict

    p = 4
    params = AttrDict({
        "p":p, 
        "emb_dim" : 100, 
        "hidden_dim" : 105,  
        "regression" : False,
        "use_wandb" : False,
        "early_stopping_grokking" : "",
        "max_epochs" : 10
    })

    bs = 4
    x = torch.zeros(size=(bs, 1, 28, 28), dtype=torch.float)
    y = torch.zeros(size=(bs,), dtype=torch.long)
    print(y.shape)

    model = Model(params)
    tensor = model(x)
    print(tensor.shape, model.criterion(input = tensor, target=y), (tensor.argmax(dim=-1) == y).float().mean())