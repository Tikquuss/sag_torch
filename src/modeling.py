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
def make_mlp(l, act=nn.LeakyReLU(), tail=[], bias=True):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*(sum(
        [[nn.Linear(i, o, bias=bias)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

class MLP(nn.Module):
    def __init__(self, l, act=nn.LeakyReLU(), tail=[], bias = True, dropout=0.0):
        super(MLP, self).__init__()
        self.net = make_mlp(l, act, tail = tail + [nn.Dropout(dropout)], bias=bias)
    def forward(self, x):
        """
        x: (bs, _)
        """
        return self.net(x) # (bs, _)

# Text
def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

class TextModel(nn.Module):
    def __init__(self, p, operator, hidden_dim, n_class, dropout=0.0, E_factor = 1.0, pad_index=None):
        super(TextModel, self).__init__()
        self.embeddings = Embedding(p, hidden_dim[0], padding_idx=pad_index)
        self.embeddings_dropout = nn.Dropout(dropout)
        self.mlp = make_mlp(hidden_dim + [n_class])
        self.mlp_dropout = nn.Dropout(dropout)
        self.operator = operator 
        self.E_factor = E_factor

    def forward(self, x):
        """
        Inputs: `x`, LongTensor(bs, 2), containing word indices
        """
        a, b = x[...,0], x[...,1] # (bs,)
        #a, b = a.unsqueeze(1), b.unsqueeze(1) # (bs, 1)
        E_a = self.embeddings_dropout(self.embeddings(a)) # (bs, emb_dim)
        E_b = self.embeddings_dropout(self.embeddings(b)) # (bs, emb_dim)
        E = (E_a + E_b) #if self.operator == "+" else E_a*E_b
        E = E / self.E_factor
        tensor = self.mlp_dropout(self.mlp(E)).squeeze() # (bs,) if regression, (bs, 2*(p - 1)+1) if classification
        return tensor

# Images

def get_out_dim_conv(h_in, w_in, kernel_size, stride, padding, dilation):
    h_out = math.floor((h_in + 2 * padding - dilation*(kernel_size - 1) - 1) / stride + 1)
    w_out = math.floor((w_in + 2 * padding - dilation*(kernel_size - 1) - 1) / stride + 1)
    return h_out, w_out

class CNNNet(nn.Module):
    def __init__(self, c_in, h_in, w_in, c_out : List, kernel_size, hidden_dim : List, 
    n_class, kernel_size_maxPool=2, dropout=0.0,
    stride=1, padding=0, dilation=1
    ):
        super(CNNNet, self).__init__()

        assert len(c_out) >= 1
        assert len(hidden_dim) >= 1

        stride_maxPool=kernel_size_maxPool

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
        h_out, w_out = get_out_dim_conv(h_in, w_in, kernel_size, stride, padding, dilation)
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
        return x
        #return F.softmax(x, dim=-1)
        #return F.log_softmax(x, dim=-1)

class ResNet(nn.Module):
    def __init__(self, c_in, h_in, w_in, 
                 hidden_dim : List, 
                 n_class, 
                 c_out : List = [64,128,128,256,512,512],
                 kernel_size = 3, 
                 kernel_size_maxPool=2, 
                 stride=1,
                 padding=1,
                 padding_maxPool=0,
                 dilation=1,
                 dropout=0.0,
                 ):
        super(ResNet, self).__init__()
        
        assert len(c_out) == 6
        assert len(hidden_dim) >= 1
        
        stride_maxPool=kernel_size_maxPool

        i = 0
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in, 
                out_channels=c_out[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm2d(c_out[i]),
            nn.ReLU(inplace=True)
        )
        h_out, w_out = get_out_dim_conv(h_in, w_in, kernel_size, stride, padding, dilation)

        i+=1
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=c_out[i-1], 
                out_channels=c_out[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm2d(c_out[i]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size_maxPool, stride_maxPool, padding_maxPool, dilation)
        )
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size, stride, padding, dilation)
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size_maxPool, stride_maxPool, padding_maxPool, dilation)

        i+=1
        self.res1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=c_out[i-1], 
                    out_channels=c_out[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation
                ),
                nn.BatchNorm2d(c_out[i]),
                nn.ReLU(inplace=True)
            ), 
            nn.Sequential(
                nn.Conv2d(
                    in_channels=c_out[i-1], 
                    out_channels=c_out[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation
                ),
                nn.BatchNorm2d(c_out[i]),
                nn.ReLU(inplace=True))
        )
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size, stride, padding, dilation)
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size, stride, padding, dilation)
        
        i+=1
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=c_out[i-1], 
                out_channels=c_out[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm2d(c_out[i]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size_maxPool, stride_maxPool, padding_maxPool, dilation)
        )
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size, stride, padding, dilation)
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size_maxPool, stride_maxPool, padding_maxPool, dilation)

        i+=1
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=c_out[i-1], 
                out_channels=c_out[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm2d(c_out[i]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size_maxPool, stride_maxPool, padding_maxPool, dilation)
        )
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size, stride, padding, dilation)
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size_maxPool, stride_maxPool, padding_maxPool, dilation)

        i+=1
        self.res2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=c_out[i-1], 
                    out_channels=c_out[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation
                ),
                nn.BatchNorm2d(c_out[i]),
                nn.ReLU(inplace=True)
            ), 
            nn.Sequential(
                nn.Conv2d(
                    in_channels=c_out[i-1], 
                    out_channels=c_out[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation
                ),
                nn.BatchNorm2d(c_out[i]),
                nn.ReLU(inplace=True))
        )
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size, stride, padding, dilation)
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size, stride, padding, dilation)

        kernel_size_maxPool=4
        self.classifier = [nn.MaxPool2d(kernel_size_maxPool, stride_maxPool, padding_maxPool, dilation)]
        h_out, w_out = get_out_dim_conv(h_out, w_out, kernel_size_maxPool, stride_maxPool, padding_maxPool, dilation)
        

        self.classifier.append(nn.Flatten())
        hidden_dim = hidden_dim + [n_class]
        self.classifier.append(nn.Linear(c_out[-1]*h_out*w_out, hidden_dim[0]))

        for i in range(1, len(hidden_dim)) :
            self.classifier.append(nn.Dropout(p=dropout))
            self.classifier.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
        self.classifier.append(nn.Dropout(p=dropout))
        
        self.classifier = nn.Sequential(*self.classifier)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.classifier(x)        
        return x

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
    
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters(params) 

        # Important: This property activates manual optimization.
        self.automatic_optimization = "sag" not in self.hparams.optimizer

        # model and loss
        regression = self.hparams.data_infos["task"] == "regression"
        self.hparams.hidden_dim = [h for h in self.hparams.hidden_dim if h!=0]
        if self.hparams.dataset_name in TORCH_SET :
            if not self.hparams.use_resnet :
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
            else :
                self.backbone = ResNet(
                    self.hparams.data_infos["c_in"], 
                    self.hparams.data_infos["h_in"], 
                    self.hparams.data_infos["w_in"],
                    hidden_dim = self.hparams.hidden_dim, 
                    n_class = self.hparams.data_infos["n_class"], 
                    c_out = [64,128,128,256,512,512],
                    kernel_size=3, 
                    kernel_size_maxPool=2, 
                    stride=1,
                    padding=1,
                    padding_maxPool=0,
                    dilation=1,
                    dropout=self.hparams.dropout
                )
        elif self.hparams.dataset_name in SKLEAN_SET:
            self.backbone = MLP(
                l = [self.hparams.data_infos["c_in"]] + self.hparams.hidden_dim + [self.hparams.data_infos["n_class"]],
                act=nn.LeakyReLU(), 
                tail=[],
                dropout = self.hparams.dropout
            )
        elif self.hparams.dataset_name == "multi_scale_feature":
            self.backbone = MLP(
                l = [self.hparams.data_infos["c_in"]] + self.hparams.hidden_dim + [self.hparams.data_infos["n_class"]],
                act=nn.LeakyReLU(), 
                tail=[],
                bias=False,
                dropout = self.hparams.dropout
            )
        elif "arithmetic" in self.hparams.dataset_name :
            self.backbone = TextModel(
                self.hparams.data_infos["p"], 
                self.hparams.data_infos["operator"], 
                self.hparams.hidden_dim, 
                self.hparams.data_infos["n_class"], 
                dropout=self.hparams.dropout, 
                E_factor = 1.0
            )
            
        if not regression :
            #self.criterion = torch.nn.NLLLoss() if self.hparams.dataset_name in TORCH_SET else nn.CrossEntropyLoss()
            self.criterion = nn.CrossEntropyLoss() 
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
            logger.info("init_y_i")
            f = '%s_%s_%s_%s'%(
                self.hparams.optimizer, self.hparams.train_batch_size, self.hparams.val_batch_size, self.hparams.train_pct
            )
            opt_path = get_hash_path(self.hparams, f, prefix="optim", suffix="")
            if os.path.isfile(opt_path) :
                optimizer = configure_optimizers(parameters, self.hparams.optimizer, self.hparams.lr_scheduler)["optimizer"]
                optimizer.load_state_dict(torch.load(opt_path))
                optim_scheduler["optimizer"].init_grad(self.device, optimizer=optimizer)
            else :
                batch_mode = optim_scheduler["optimizer"].batch_mode
                batch_size = self.hparams.train_batch_size if batch_mode else 1
                train_loader = torch.utils.data.DataLoader(self.hparams.train_dataset, batch_size=batch_size, shuffle=False)
                optim_scheduler["optimizer"].init_grad(self.device, train_loader=train_loader, model=self)
                torch.save(optim_scheduler["optimizer"].state_dict(), opt_path)

        return optim_scheduler

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

        self.log('train_loss', loss, prog_bar=True)
        output = {"loss" : loss}
        if not (self.hparams.data_infos["task"] == "regression") : 
            acc = (tensor.argmax(dim=-1) == y).float().mean() * 100
            output["train_acc"] = acc
            self.log('train_acc', acc, prog_bar=True)

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
                    #sch.step(self.trainer.callback_metrics[sch.monitor])
                    sch.step(loss.item() if "loss" in self.hparams.lr_scheduler else acc.item())
                else:
                    sch.step()

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
        logs = {
            "train_loss": loss,
            "train_epoch" : self.current_epoch,
        }

        if 'train' in self.es_metric : logs["e_step"] = self.increase_es_limit(logs)

        if self.hparams.data_infos["task"] == "regression" : 
            memo_condition = round(loss.item(), 10) == 0.0
        else : 
            accuracy = torch.stack([x["train_acc"] for x in outputs]).mean()
            logs["train_acc"] = accuracy
            memo_condition = accuracy >= 99.0

        self.memorization = self.memorization or memo_condition
        if memo_condition : self.memo_epoch = min(self.current_epoch, self.memo_epoch)
    
        schedulers = self.lr_schedulers()
        if schedulers is not None :
            try : scheduler = schedulers[0]
            except TypeError: scheduler = schedulers # 'xxx' object is not subscriptable
            param_groups = scheduler.optimizer.param_groups
            logs["lr"] = param_groups[0]["lr"]

        for k, v in logs.items() : self.log(k, v, prog_bar=True)
        if self.use_wandb: wandb.log(logs)

    def validation_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        """
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {
            "val_loss": loss,    
            "val_epoch": self.current_epoch,
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
        db_data = {k : v for k, v in db_data.items() if k != "classes" and (v is not None) and type(v) != str}
        db_data = {k : v for k, v in db_data.items() if type(v) != bool}
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