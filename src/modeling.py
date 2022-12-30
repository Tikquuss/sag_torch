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

from .optim import configure_optimizers

possible_metrics = ["%s_%s"%(i, j) for i, j in itertools.product(["train", "val"], ["acc", "loss"])]

class Net(nn.Module):
    def __init__(self, dropout=0.5):
        super(Net, self).__init__()
        kernel_size=5
        hidden_dim=50
        n_class=10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(320, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) # (bs, n_class)
        return F.log_softmax(x, dim=-1)

class Model(pl.LightningModule):
    """
    params : 
        - p (int), emb_dim (int), hidden_dim (int),  n_layers (int), regression (bool), modular (bool)
        - pad_index (int, optional, None), use_wandb (int, optional, False)
        - representation_lr and decoder_lr (float, optional, 1e-3), weight_decay (float, optional, 0) 
        - factor (float, optional, 0.2), patience (float, optional, 20), min_lr (float, optional, 5e-5)
    """
    def __init__(self, params):
        """
        Transformer model 
        """
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters(params) 

        self.backbone = Net()

        if self.hparams.regression : self.criterion = nn.MSELoss()
        else :
            #self.criterion = nn.CrossEntropyLoss() 
            self.criterion = torch.nn.NLLLoss()

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
        self.es_metric = early_stopping_grokking.get("metric", "val_loss" if self.hparams.regression else "val_acc") 
        assert self.es_metric in possible_metrics
        self.es_metric_threshold = early_stopping_grokking.get("metric_threshold", 0.0 if 'loss' in self.es_metric else 99.0) 
        self.es_mode = (lambda s : "min" if 'loss' in s else 'max')(self.es_metric)
        self.es_step = 0
        self.reached_limit = False

    def configure_optimizers(self):
        parameters = [
            {'params': self.backbone.parameters(), 'lr': self.hparams.get("lr", 1e-3)}, 
        ]
        return configure_optimizers(parameters, self.hparams.optimizer, self.hparams.lr_scheduler)

    def forward(self, x):
        """
        Inputs: `x`, Tensor(bs, 1, 28, 28), 
        """
        return self.backbone(x) # (bs,)
    
    def _get_loss(self, batch):
        """
        Given a batch of data, this function returns the  loss (MSE or CEL)
        """
        x, y = batch # We do not need the labels
        tensor = self.forward(x)
        loss = self.criterion(input = tensor, target=y)
        return loss, tensor, y
    
    def training_step(self, batch, batch_idx):
        loss, tensor, y = self._get_loss(batch)  
        self.log('train_loss', loss, prog_bar=True)
        output = {"loss" : loss}
        if not self.hparams.regression : 
            acc = (tensor.argmax(dim=-1) == y).float().mean() * 100
            output["train_acc"] = acc
            self.log('train_acc', acc, prog_bar=True)
        return output 
    
    def validation_step(self, batch, batch_idx):
        loss, tensor, y = self._get_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
        output = {'val_loss' : loss}
        if not self.hparams.regression : 
            acc = (tensor.argmax(dim=-1) == y).float().mean() * 100
            output["val_acc"] = acc
            self.log('val_acc', acc, prog_bar=True)
        return  output 
    
    def test_step(self, batch, batch_idx):
        loss, tensor, y = self._get_loss(batch)
        #self.log('test_loss', loss, prog_bar=True)
        output = {'test_loss' : loss}
        if not self.hparams.regression : 
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

        if self.hparams.regression : memo_condition = round(loss.item(), 10) == 0.0
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

        if self.hparams.regression : comp_condition = round(loss.item(), 10) == 0.0
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