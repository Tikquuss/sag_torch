import os
from typing import List, Union, Dict
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision
import pytorch_lightning as pl
from loguru import logger

DATA_PATH="../data"
#os.makedirs(DATA_PATH, exist_ok=True)

class DatasetWithIndexes(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

def get_dataloader(x, y, train_pct, batch_size, num_workers=0):
    """We define a data constructor that we can use for various purposes later."""
  
    dataset = TensorDataset(x, y)
    dataset = DatasetWithIndexes(dataset)
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

def cut_dataset(dataset, pct):
    n = len(dataset)
    size = pct * n // 100
    print(f"size, n-size : {size}, {n-size}")
    remaining, _ = torch.utils.data.random_split(dataset, [size, n - size])
    return remaining
    
class LMLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name : str,
        train_batch_size: int,
        val_batch_size: int,
        train_pct: int = 100,
        val_pct: int = 100,
        data_path: str = DATA_PATH,
        num_workers: int = 0,  
    ):
        super(LMLightningDataModule, self).__init__()
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.data_path = data_path
        self.num_workers = num_workers
        self.prepare_data()
        
    def prepare_data(self):
        logger.info(f"Dataset {self.dataset_name} loading....")
        os.makedirs(self.data_path, exist_ok=True)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]
            )
        if self.dataset_name == "mnist" :
            self.train_dataset = torchvision.datasets.MNIST(self.data_path, train=True, download=True, transform = transform)
            self.val_dataset =  torchvision.datasets.MNIST(self.data_path, train=False, download=True, transform = transform)
            c_in, h_in, w_in, n_class = 1, 28, 28, 10
        elif self.dataset_name == "fashion_mnist" :
            self.train_dataset = torchvision.datasets.FashionMNIST(self.data_path, train=True, download=True, transform = transform)
            self.val_dataset =  torchvision.datasets.FashionMNIST(self.data_path, train=False, download=True, transform = transform)
            c_in, h_in, w_in, n_class = 1, 28, 28, 10
        elif self.dataset_name == "cifar10" :
            self.train_dataset = torchvision.datasets.CIFAR10(self.data_path, train=True, download=True, transform = transform)
            self.val_dataset =  torchvision.datasets.CIFAR10(self.data_path, train=False, download=True, transform = transform)
            c_in, h_in, w_in, n_class = 3, 32, 32, 10
        elif self.dataset_name == "cifar100" :
            self.train_dataset = torchvision.datasets.CIFAR10(self.data_path, train=True, download=True, transform = transform)
            self.val_dataset =  torchvision.datasets.CIFAR10(self.data_path, train=False, download=True, transform = transform)
            c_in, h_in, w_in, n_class = 3, 32, 32, 100
        else :
            raise Exception("Unknown dataset : %s" % self.dataset_name)

        if 0 < self.train_pct < 100 :
            self.train_dataset = cut_dataset(self.train_dataset, pct=self.train_pct)
        if 0 < self.val_pct < 100 :
            self.val_dataset = cut_dataset(self.val_dataset, pct=self.val_pct)

        self.train_dataset = DatasetWithIndexes(self.train_dataset)
        self.val_dataset = DatasetWithIndexes(self.val_dataset)

        train_size = len(self.train_dataset)
        val_size = len(self.val_dataset)
        self.train_batch_size = min(self.train_batch_size, train_size)
        self.val_batch_size = min(self.val_batch_size, val_size)
        self.data_infos = {
            "c_in" : c_in, "h_in" : h_in, "w_in" : w_in, "n_class" : n_class,
            "train_batch_size" : self.train_batch_size, "val_batch_size" : self.val_batch_size, 
            "train_size":train_size, "val_size":val_size, 
            "train_n_batchs":len(self.train_dataloader()), "val_n_batchs":len(self.val_dataloader())
        }

        logger.info(self.data_infos)

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