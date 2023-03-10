import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

from .multi_scale_feature import DatasetWithIndexes, get_modulation_matrix_multi_singular

def iid_normal(dim, sample_shape, mu, sigma) :
    mean = torch.zeros(dim) + mu
    cov = torch.eye(dim) * sigma
    dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
    #dist = torch.distributions.normal.Normal(loc=mean, scale=torch.diagonal(cov))
    return dist.sample(sample_shape = (sample_shape,))

def teacher(N, P, M, out_dim = 1, sigma_noise = 1.0, g = lambda x : x):
    """
    N : input dimension 
    P : number of sample
    M : hidden dimension
    out_dim : output dim
    sigma_noise : noise standard deviation
    g : activation function
    """
    mu_x, sigma_x = 0.0, 1.0 # data
    mu_w, sigma_w = 0.0, 1.0 # feature map
    mu_v, sigma_v = 0.0, 1.0 # output layer
    #mu_noise, sigma_noise = 0.0, 1.0 # noise
    mu_noise = 1.0 # noise

    # data
    x = iid_normal(dim=N, sample_shape=P, mu=mu_x, sigma=sigma_x) # P x N
    # weigths
    w = iid_normal(dim=N, sample_shape=M, mu=mu_w, sigma=sigma_w) # M x N
    v = iid_normal(dim=out_dim, sample_shape=M, mu=mu_v, sigma=sigma_v) # M x out_dim
    # noise
    noise = iid_normal(dim=out_dim, sample_shape=P, mu=mu_noise, sigma=sigma_noise) # P x out_dim
    noise = noise.squeeze()

    hidden_units = g(w @ x.T / np.sqrt(N)) # M x P
    y = v.T @ hidden_units # out_dim x P
    y = y.squeeze().T # P x out_dim
    
    return x, y, noise

def get_data(seed, train_size, val_size, N, M, noise, out_dim = 1, k = None, g = lambda x : x):
    np.random.seed(seed)
    torch.manual_seed(0)

    P = train_size+val_size
    x, y, noise_vec = teacher(N, P, M, out_dim = out_dim, sigma_noise = 1.0, g = g)

    if k is not None :
        F_matrix, _, _, _ = get_modulation_matrix_multi_singular(N, k) # NxN
        x = x @ torch.from_numpy(F_matrix).float() # P x N

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = val_size/P, random_state = seed)

    y_train = y_train + noise*noise_vec[:train_size]
    # test data is noiseless
    y_test = y_test + 0*noise_vec[train_size:]
    
    return x_train, y_train, x_test, y_test

def get_dataloader(train_size, val_size, N, M, k, out_dim = 1, g = lambda x : x, noise = 0.0, seed = 100, task = "regression", include_indexes = False, train_batch_size = None, val_batch_size = None, num_workers=0, return_just_set = True):
    """
    # train_size, val_size : number of training/validation examples
    # N: number of total dimensions
    # k: conditions numbers
    # noise: standard deviation of the noise added to the teacher output
    """
    x_train, y_train, x_test, y_test = get_data(seed, train_size, val_size, N, M, noise, out_dim, g=g, k = k)

    if task == "regression" : y_train, y_test = y_train.float(), y_test.float()
    else : y_train, y_test = y_train.long(), y_test.long()

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
        "train_batch_size" : min(train_batch_size, train_size), "val_batch_size" : min(val_batch_size, val_size), 
        "train_size":train_size, "val_size":val_size, 
        "train_n_batchs":len(train_loader), "val_n_batchs":len(val_loader)
    }

    return train_loader, val_loader, data_infos