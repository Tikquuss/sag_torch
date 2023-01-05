import os
from typing import List, Union, Dict
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler, RandomSampler
import torchvision
import pytorch_lightning as pl
from sklearn import datasets as sk_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger
import math

DATA_PATH="../data"
#os.makedirs(DATA_PATH, exist_ok=True)

TORCH_SET = ["mnist", "fashion_mnist", "cifar10", "cifar100",]
SKLEAN_SET = ["wine", "boston", "iris", "diabete", "digits", "linnerud"]
OTHER_SET = ["arithmetic"]
DATA_SET = TORCH_SET + SKLEAN_SET + OTHER_SET

from .utils import str2dic, bool_flag

class DatasetWithIndexes(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

def get_dataloader(x, y, train_pct, include_indexes = False, train_batch_size = None, val_batch_size = None, num_workers=0, return_just_set = True):
    """We define a data constructor that we can use for various purposes later."""
  
    dataset = TensorDataset(x, y)
    if include_indexes :
        dataset = DatasetWithIndexes(dataset)
    n = len(dataset)
    train_size = train_pct * n // 100
    val_size = n - train_size
    print(f"train_size, val_size : {train_size}, {val_size}")

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    if return_just_set :
        return train_set, val_set

    assert train_batch_size is not None
    assert val_batch_size is not None
    train_loader = DataLoader(train_set, batch_size=min(train_batch_size, train_size), shuffle=True, drop_last=False, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=min(val_batch_size, val_size), shuffle=False, drop_last=False, num_workers=num_workers)
    dataloader = DataLoader(dataset, batch_size=min(train_batch_size, n), shuffle=False, drop_last=False, num_workers=num_workers)

    data_infos = {
        "train_batch_size" : min(train_batch_size, train_size), "val_batch_size" : min(val_batch_size, val_size), 
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

def get_arithmetic_set(p, regression, operator="+", ij_equal_ji = True, modular = True):
    """We define a data constructor that we can use for various purposes later."""
    assert operator in ["+", "*"]
    if ij_equal_ji :
        x = []
        for i in range(p) :
          for j in range(i, p) :
              x.append([i, j])
        x = torch.LongTensor(x) # (p*(p+1)/2, 2)
    else :
        ij = torch.arange(p) # (p,)
        x = torch.cartesian_prod(ij, ij) # (p^2, 2)
    y = x.sum(1) if operator=="+" else x.prod(1) # (p*(p+1)/2,) if ij_equal_ji, else # (p^2,)
    if modular : y = torch.remainder(y, p)
    if regression : y = y.float() 
    return x, y

class LMLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name : str,
        train_batch_size: int,
        val_batch_size: int,
        train_pct: int = 100,
        val_pct: int = 100,
        use_sampler : bool = False, 
        data_path: str = DATA_PATH,
        num_workers: int = 0,  
    ):
        super(LMLightningDataModule, self).__init__()
        assert dataset_name in DATA_SET or "arithmetic" in dataset_name
        if dataset_name in SKLEAN_SET or "arithmetic" in dataset_name : assert 0 < train_pct < 100
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.data_path = data_path
        self.num_workers = num_workers

        self.use_sampler = use_sampler
        self.sampler = None

        self.prepare_data()
        
    def prepare_data(self):
        logger.info(f"Dataset {self.dataset_name} loading....")
        os.makedirs(self.data_path, exist_ok=True)
        h_in, w_in = 0, 0
        tmp = {}
        if self.dataset_name == "mnist" :
            # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457?u=pascal_notsawo
            # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/31?u=pascal_notsawo
            """
            data = train_dataset.data.float()
            mean = np.round(data.mean(axis=(0,1,2))/255,4)
            std = np.round(data.std(axis=(0,1,2))/255,4)
            """
            mean, std = 0.1307000070810318, 0.30809998512268066
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean,), (std,))])
            self.train_dataset = torchvision.datasets.MNIST(self.data_path, train=True, download=True, transform = transform)
            self.val_dataset = torchvision.datasets.MNIST(self.data_path, train=False, download=True, transform = transform)
            c_in, h_in, w_in, n_class = 1, 28, 28, 10
            task = "classification"
            classes = tuple(range(10))
        elif self.dataset_name == "fashion_mnist" :
            mean, std = 0.28600001335144043, 0.3529999852180481
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean,), (std,))])
            self.train_dataset = torchvision.datasets.FashionMNIST(self.data_path, train=True, download=True, transform = transform)
            self.val_dataset =  torchvision.datasets.FashionMNIST(self.data_path, train=False, download=True, transform = transform)
            c_in, h_in, w_in, n_class = 1, 28, 28, 10
            task = "classification"
            classes = tuple(range(10))
        elif self.dataset_name == "cifar10" :
            #mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
                # https://medium.com/mlearning-ai/cifar10-image-classification-in-pytorch-e5185176fbef
                #torchvision.transforms.RandomResizedCrop(224), # h_in = w_in = 224
                #torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                #torchvision.transforms.RandomHorizontalFlip(p=0.5)
                ])
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)])
            self.train_dataset = torchvision.datasets.CIFAR10(self.data_path, train=True, download=True, transform = train_transform)
            self.val_dataset =  torchvision.datasets.CIFAR10(self.data_path, train=False, download=True, transform = transform)
            c_in, h_in, w_in, n_class = 3, 32, 32, 10
            task = "classification"
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        elif self.dataset_name == "cifar100" :
            #mean, std = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
                # https://medium.com/mlearning-ai/cifar10-image-classification-in-pytorch-e5185176fbef
                #torchvision.transforms.RandomResizedCrop(224), # h_in = w_in = 224
                torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                torchvision.transforms.RandomHorizontalFlip(p=0.5)])
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)])
            self.train_dataset = torchvision.datasets.CIFAR10(self.data_path, train=True, download=True, transform = train_transform)
            self.val_dataset =  torchvision.datasets.CIFAR10(self.data_path, train=False, download=True, transform = transform)
            c_in, h_in, w_in, n_class = 3, 32, 32, 100
            task = "classification"
            # TODO : https://www.cs.toronto.edu/~kriz/cifar.html
            classes = tuple(range(100))
        elif self.dataset_name == "wine" :
            # recognize the wine class given the features like the amount of alcohol, magnesium, phenol, color intensity, etc
            dataset = sk_dataset.load_wine()
            classes = tuple(dataset["target_names"])
            c_in, n_class = 13, len(classes)
            task = "classification"
        elif self.dataset_name == "boston" :
            # houses in Boston like the crime rate, nitric oxides concentration, number of rooms, distances to employment centers, tax rates, etc. 
            # The output feature is the median value of homes.
            dataset = sk_dataset.load_boston()
            c_in, n_class = 13, 1
            task = "regression"
            classes = None
        elif self.dataset_name == "iris" :
            # It contains sepal and petal lengths and widths for three classes of plants
            dataset = sk_dataset.load_iris()
            classes = tuple(dataset["target_names"])
            c_in, n_class = 4, len(classes)
            task = "classification"
        elif self.dataset_name == "diabete" :
            # the diabetes dataset (regression)
            dataset = sk_dataset.load_diabetes()
            c_in, n_class = 10, 1
            task = "regression"
            classes = None
        elif self.dataset_name == "digits" :
            # Load and return the digits dataset (classification)
            dataset = sk_dataset.load_digits()
            classes = tuple(dataset["target_names"])
            c_in, n_class = 64, len(classes)
            task = "classification"
        elif self.dataset_name == "linnerud" :
            # Load and return the physical exercise Linnerud dataset (regression)
            dataset = sk_dataset.load_linnerud()
            c_in, n_class = 3, 3
            task = "regression"
            classes = None
        elif "arithmetic" in self.dataset_name :
            #"arithmetic,op=+,p=200,reg=False,mod=True,ijeqji=True"
            s = self.dataset_name.split("arithmetic,")[1]
            s = str2dic(s)
            #assert 'p' in s.keys()
            op, p, reg, mod, ijeqji = s["op"], int(s["p"]), bool_flag(s["reg"]), bool_flag(s["reg"]), bool_flag(s["ijeqji"])
            tmp = {"p" : p, "regression" : reg, "operator" : op, "ij_equal_ji" : ijeqji, "modular" : mod}
            #x, y = get_arithmetic_set(p, regression=reg, operator=op, ij_equal_ji = ijeqji, modular = mod)
            x, y = get_arithmetic_set(**tmp)
            self.train_dataset, self.val_dataset = get_dataloader(
                x, y,
                train_pct=self.train_pct, 
                num_workers=self.num_workers
            )
            n_class = 2*(p-1)+1 if op == "+" else (p-1)**2+1
            n_class = p if mod else n_class
            n_class = 1 if reg else n_class
            if reg :
                classes = None
                task = "regression" if reg else "classification"
            else :
                classes = tuple(range(n_class))
                task = "regression" if reg else "classification"
            c_in = 0
        else :
            # TODO : https://scikit-learn.org/stable/datasets/real_world.html
            raise Exception("Unknown dataset : %s" % self.dataset_name)

        if self.dataset_name in SKLEAN_SET :
            x = dataset["data"]
            # Scale data to have mean 0 and variance 1 
            # which is importance for convergence of the neural network
            scaler = StandardScaler()
            x = scaler.fit_transform(x)

            y = torch.from_numpy(dataset["target"])
            if task == "regression" : y = y.float()
            else : y = y.long()

            self.train_dataset, self.val_dataset = get_dataloader(
                torch.from_numpy(x).float(), y,
                train_pct=self.train_pct, 
                num_workers=self.num_workers
            )

        cond = (self.dataset_name not in SKLEAN_SET) or ("arithmetic" not in self.dataset_name) 
        if cond and (0 < self.train_pct < 100) :
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
            "classes" : classes, "task" : task,
            "train_batch_size" : self.train_batch_size, "val_batch_size" : self.val_batch_size, 
            "train_size":train_size, "val_size":val_size, 
            "train_n_batchs":len(self.train_dataloader()), "val_n_batchs":len(self.val_dataloader())
        }
        for k, v in tmp.items() : self.data_infos[k] = v

        logger.info(self.data_infos)
        for k, v in self.data_infos.items() : logger.info(str(k) + " --> " + str(v))

        if self.use_sampler :
            weights = torch.ones(train_size) / train_size 
            num_samples = train_size
            #num_samples =  math.ceil(train_size / self.train_batch_size) * max_epochs
            self.sampler = WeightedRandomSampler(weights = weights, num_samples=num_samples, replacement=True, generator=None)
            #self.sampler = RandomSampler(data_source=self.train_dataset, replacement=True, num_samples=num_samples, generator=None)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=not self.use_sampler,
            sampler=self.sampler
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