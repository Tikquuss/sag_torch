import os
from typing import List, Union, Dict
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import pytorch_lightning as pl
from loguru import logger

DATA_PATH="../data"
#os.makedirs(DATA_PATH, exist_ok=True)

def get_dataloader(x, y, train_pct, batch_size, num_workers=4):
    """We define a data constructor that we can use for various purposes later."""
  
    dataset = TensorDataset(x, y)
    n = len(dataset)
    train_size = train_pct * n // 100
    val_size = n - train_size
    print(f"train_size, val_size : {train_size}, {val_size}")

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=min(batch_size, train_size), shuffle=True, drop_last=False, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=min(batch_size, val_size), shuffle=False, drop_last=False, num_workers=num_workers)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, n), shuffle=False, drop_last=False, num_workers=num_workers)

    data_infos = {
        "train_batch_size" : min(batch_size, train_size), "val_batch_size" : min(batch_size, val_size), 
        "train_size":train_size, "val_size":val_size, 
        "train_n_batchs":len(train_loader), "val_n_batchs":len(val_loader)
    }

    return train_loader, val_loader, dataloader, data_infos


class LMLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name : str,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int = 0,  
    ):
        super(LMLightningDataModule, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.prepare_data()
        
    def prepare_data(self):
        logger.info(f"Dataset {self.dataset_name} loading....")
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]
            )
        if self.dataset_name == "mnist" :
            self.train_dataset = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform = transform)
            self.val_dataset =  torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform = transform)
        elif self.dataset_name == "fashion_mnist" :
            self.train_dataset = torchvision.datasets.FashionMNIST(DATA_PATH, train=True, download=True, transform = transform)
            self.val_dataset =  torchvision.datasets.FashionMNIST(DATA_PATH, train=False, download=True, transform = transform)
        elif self.dataset_name == "cifar10" :
            self.train_dataset = torchvision.datasets.CIFAR10(DATA_PATH, train=True, download=True, transform = transform)
            self.val_dataset =  torchvision.datasets.CIFAR10(DATA_PATH, train=False, download=True, transform = transform)

        train_size = len(self.train_dataset)
        val_size = len(self.val_dataset)
        self.train_batch_size = min(self.train_batch_size, train_size)
        self.val_batch_size = min(self.val_batch_size, val_size)
        self.data_infos = {
            "train_batch_size" : self.train_batch_size, "val_batch_size" : self.val_batch_size, 
            "train_size":train_size, "val_size":val_size, 
            "train_n_batchs":len(self.train_dataloader()), "val_n_batchs":len(self.val_dataloader())
        }

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":

    data_module = LMLightningDataModule(
        dataset_name = "mnist",
        train_batch_size = 512,
        val_batch_size = 1000,
        num_workers = 0,
        
    )
    print(data_module.data_infos)
    
    x, y = next(iter(data_module.train_dataloader()))
    print(x.shape, y.shape)