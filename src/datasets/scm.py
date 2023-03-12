import torch
import torch.nn.functional as F
import torch.distributions as Dist
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

from .multi_scale_feature import DatasetWithIndexes, get_modulation_matrix_multi_singular

def iid_normal(dim : int, sample_shape : Tuple, mu : float, sigma : float) :
    dist = Dist.MultivariateNormal(loc=torch.zeros(dim)+mu, covariance_matrix=torch.eye(dim)*sigma)
    #dist = Dist.Normal(loc=mu, scale=torch.ones(dim)*sigma)
    return dist.sample(sample_shape = sample_shape)

def iid_mixture_normal(dim : int, sample_shape : Tuple, mu : List, sigma : List, weights : List) :
    n_components = len(weights)
    assert len(mu) == len(sigma) == n_components
    dist = MixtureSameFamily(
        mixture_distribution = Dist.Categorical(probs=torch.tensor(weights)),
        component_distribution = Dist.Independent(Dist.Normal(
            loc=torch.tensor([[mu[i]]*dim for i in range(n_components)]), # (n_components, dim)
            scale=torch.tensor([[sigma[i]]*dim for i in range(n_components)]), # (n_components, dim)
        ), 1),
    )
    return dist.sample(sample_shape = sample_shape)

def get_weights(N, M, out_dim = 1,
            mu_w = 0.0, sigma_w = 1.0, # feature map
            mu_v = 0.0, sigma_v = 1.0, # output layer
            ):
    """
    N : input dimension 
    P : number of sample
    M : hidden dimension
    out_dim : output dim
    """
    # weigths
    w = iid_normal(dim=N, sample_shape=(M,), mu=mu_w, sigma=sigma_w) # (M, N)
    v = iid_normal(dim=out_dim, sample_shape=(M,), mu=mu_v, sigma=sigma_v) # (M, out_dim)
    return w, v

def forward(w, v, x, N, g, denom = None):
    if denom is None : denom = np.sqrt(N)
    # hidden_units = w @ x.T / denom # (?, N) x (N, P) = (?, P)
    # if g is not None : activation = g(hidden_units) # (?, P)
    # else : activation = hidden_units # (?, P)
    # y = v.T @ activation # (out_dim, ?) x (?, P) = (out_dim, P)
    # y = y.T.squeeze() # (P, out_dim) or (P,) if out_dim=1

    hidden_units = x @ w.T / denom # (P, N) x (N, ?) = (P, ?)
    #hidden_units = torch.nn.functional.linear(x, w, bias=None) / np.sqrt(N) # (P, N) x (N, ?) = (P, ?)
    if g is not None : activation = g(hidden_units) # (P, ?)
    else : activation = hidden_units # (P, ?)
    y = activation @ v # (P, ?) x (?, out_dim) = (P, out_dim)
    #y = y.squeeze() # (P, out_dim) or (P,) if out_dim=1

    return y, activation, hidden_units

def backward(y_hat, y, activation):
    grad_w = None
    delta = y_hat - y # (P, out_dim)
    grad_v = activation.unsqueeze(dim=2) @ delta.unsqueeze(dim=1) # (P, ?, 1) x (P, 1, out_dim) = (P, ?, out_dim)
    grad_v = 2 * grad_v.mean(dim=0) # (?, out_dim)
    return grad_w, grad_v

def teacher(N, M, P, out_dim = 1, g = None,
            mu_x = 0.0, sigma_x = 1.0, weights_x = None, # data
            mu_w = 0.0, sigma_w = 1.0, # feature map
            mu_v = 0.0, sigma_v = 1.0, # output layer
            mu_noise = 0.0, sigma_noise = 1.0, # noise
            scm = False
            ):
    """
    N : input dimension 
    P : number of sample
    M : hidden dimension
    out_dim : output dim
    sigma_noise : noise standard deviation
    g : activation function
    """
    # data
    if weights_x is None : x = iid_normal(dim=N, sample_shape=(P,), mu=mu_x, sigma=sigma_x) # (P, N)
    else : x = iid_mixture_normal(dim=N, sample_shape=(P,), mu=mu_x, sigma=sigma_x, weights=weights_x) # (P, N)
    # weigths 
    w, v = get_weights(N, M, out_dim, 
                       mu_w = mu_w, sigma_w = sigma_w, # feature map
                       mu_v = mu_v, sigma_v = sigma_v, # output layer
                       )
    #if scm : v = torch.ones_like(v)
    v = torch.ones_like(v)*sigma_v
    # noise
    noise = iid_normal(dim=out_dim, sample_shape=(P,), mu=mu_noise, sigma=sigma_noise) # (P, out_dim)
    #noise = noise.squeeze() # (P, out_dim)
    y, _, _ = forward(w, v, x, N, g) # (P, out_dim) 
    return x, y, noise, w, v

def binarize(y):
    return 0.0*(y>0) + 1.0*(y<=0)

def get_data(seed, train_size, val_size, N, M, out_dim = 1, 
            k = None, singular_val=1.0,
            g = None,
            mu_x = 0.0, sigma_x = 1.0, weights_x = None, # data
            mu_w = 0.0, sigma_w = 1.0, # feature map
            mu_v = 0.0, sigma_v = 1.0, # output layer
            mu_noise = 0.0, sigma_noise = 1.0, # noise
            scm = False, task = "regression"
             ):
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    P = train_size+val_size
    x, y, noise_vec, w, v = teacher(N, M, P, out_dim = out_dim, g = g,
                                    mu_x = mu_x, sigma_x = sigma_x, weights_x = weights_x, # data
                                    mu_w = mu_w, sigma_w = sigma_w, # feature map
                                    mu_v = mu_v, sigma_v = sigma_v, # output layer
                                    mu_noise = mu_noise, sigma_noise = 1.0, # noise
                                    scm = scm
                                    )

    F_matrix = torch.eye(N)
    if k is not None :
        F_matrix, _, _, _ = get_modulation_matrix_multi_singular(N, k, singular_val) # (N, N)
        F_matrix = torch.from_numpy(F_matrix).float()
    x = x @ F_matrix # (P, N)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = val_size/P, random_state = seed)
    
    y_train = y_train + sigma_noise*noise_vec[:len(y_train)]
    # test data is noiseless
    y_test = y_test + 0*noise_vec[len(y_train):]

    if "bin" in task :
        y_train = binarize(y_train)
        y_test = binarize(y_test)
    if "multi" in task :
        #y_train = torch.empty(y_train.size(0)).random_(out_dim)
        #y_test = torch.empty(y_test.size(0)).random_(out_dim)
        y_train = binarize(y_train).sum(dim=1).remainder(out_dim) 
        y_test = binarize(y_test).sum(dim=1).remainder(out_dim)

    return x_train, y_train, x_test, y_test, w, v, F_matrix

class DatasetWithIndexes(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)
    
def get_dataloader(train_size, val_size, N, M, 
    k=None, singular_val=1.0,
    out_dim = 1, g = None, 
    mu_x = 0.0, sigma_x = 1.0, weights_x = None,# data
    mu_w = 0.0, sigma_w = 1.0, # feature map
    mu_v = 0.0, sigma_v = 1.0, # output layer
    mu_noise = 0.0, sigma_noise = 1.0, # noise
    scm = False,
    seed = 100, task = "regression", include_indexes = False, train_batch_size = None, val_batch_size = None, num_workers=0, return_just_set = True):
    """
    # train_size, val_size : number of training/validation examples
    # N: number of total dimensions
    # k: conditions numbers
    # noise: standard deviation of the noise added to the teacher output
    """
    x_train, y_train, x_test, y_test, w, v, F_matrix = get_data(seed, train_size, val_size, N, M, out_dim, 
                                    k=k, singular_val=singular_val,
                                    g=g,
                                    mu_x = mu_x, sigma_x = sigma_x, weights_x = weights_x, # data
                                    mu_w = mu_w, sigma_w = sigma_w, # feature map
                                    mu_v = mu_v, sigma_v = sigma_v, # output layer
                                    mu_noise = mu_noise, sigma_noise = sigma_noise, # noise
                                    scm = scm, task = task,
                                    )
    
    data_infos = {}
    if "classification" in task :
        data_infos[f"class_train"] = {c : (y_train == c).sum().item() for c in range(out_dim+1)}
        data_infos[f"class_test"] = {c : (y_test == c).sum().item() for c in range(out_dim+1)}

    y_train, y_test = y_train.float(), y_test.float()
    #if task == "regression" : y_train, y_test = y_train.float(), y_test.float()
    #else : y_train, y_test = y_train.long(), y_test.long()
    if "multi" in task : y_train, y_test = y_train.long(), y_test.long()

    train_set = TensorDataset(x_train, y_train)
    val_set = TensorDataset(x_test, y_test)

    if include_indexes :
        train_set = DatasetWithIndexes(train_set)
        val_set = DatasetWithIndexes(val_set)

    if return_just_set :
        return train_set, val_set

    assert train_batch_size is not None
    assert val_batch_size is not None
    train_loader = DataLoader(train_set, batch_size=min(train_batch_size, train_size), shuffle=True, drop_last=False, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=min(val_batch_size, val_size), shuffle=False, drop_last=False, num_workers=num_workers)

    data_infos = {
        **data_infos, 
        **{
        "train_batch_size" : min(train_batch_size, train_size), "val_batch_size" : min(val_batch_size, val_size), 
        "train_size":train_size, "val_size":val_size, 
        "train_n_batchs":len(train_loader), "val_n_batchs":len(val_loader)
        }
    }

    return train_loader, val_loader, data_infos, w, v, F_matrix


if __name__ == "__main__":
    seed=10
    np.random.seed(seed)
    torch.manual_seed(seed)

    s={"N":800, "M":4, "K":6, "fixed_w":False, "scm":False, "g":"id", "noise":0.001, "out_dim":1}
    train_size, val_size = 100, 1000
    mu_x, sigma_x, weights_x = 0.0, 0.1, None
    mu_x, sigma_x, weights_x = [-50.0, 50.0], [1.0, 2.0], [0.6, 0.4]

    k = [10, 1]
    k = {10 : 0.2}
    k = None
    singular_val=1.0
    
    task = "regression"
    task = "classification" 
    
    train_batch_size, val_batch_size = 2**20, 2**20
    mu_w, sigma_w = 0.0, 1.0 # feature map
    mu_v, sigma_v = 0.0, 1.0 # output layer
    N, M, K, fixed_w, scm, g = s["N"], s["M"], s["K"], s["fixed_w"], s["scm"], s["g"] 
    noise, out_dim = s.get("noise", 0.0), s.get("out_dim", 1)

    train_loader, val_loader, data_infos, w_start, v_start, F_matrix = get_dataloader(
        train_size, val_size, N, M, 
        k=k, singular_val=singular_val,
        out_dim = out_dim, g=None, 
        mu_x = mu_x, sigma_x = sigma_x, weights_x=weights_x, # data
        mu_w = mu_w, sigma_w = sigma_w, # feature map
        mu_v = mu_v, sigma_v = sigma_v, # output layer
        mu_noise = 0.0, sigma_noise = noise, # noise
        scm = scm,
        seed = seed, task = task, 
        include_indexes = False, 
        train_batch_size = train_batch_size, val_batch_size = val_batch_size, 
        num_workers=0, 
        return_just_set = False
    )

    for key, val in data_infos.items()  : print(str(key) + " --> " + str(val))