import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

from .multi_scale_feature import DatasetWithIndexes, get_modulation_matrix_multi_singular
from .scm import iid_normal, binarize

def get_weights(N, out_dim = 1,
            mu_w = 0.0, sigma_w = 1.0, # w
            ):
    """
    N : input dimension 
    P : number of sample
    M : hidden dimension
    out_dim : output dim
    """
    # weigths
    w = iid_normal(dim=out_dim, sample_shape=(N,), mu=mu_w, sigma=sigma_w) # (N, out_dim)
    return w

def forward(w, x, N, g, denom = None):
    if denom is None : denom = np.sqrt(N)
    hidden_units = x @ w / denom # (P, N) x (N, out_dim) = (P, out_dim)
    if g is not None : activation = g(hidden_units) # (P, out_dim)
    else : activation = hidden_units # (P, out_dim)
    return activation, hidden_units

def backward(y_hat, y, activation):
    grad_w = None
    return grad_w

def teacher(N, P, out_dim = 1, g = None,
            mu_x = 0.0, sigma_x = 1.0, # data
            mu_w = 0.0, sigma_w = 1.0, # feature map
            mu_noise = 0.0, sigma_noise = 1.0, # noise
            ):
    """
    N : input dimension 
    P : number of sample
    out_dim : output dim
    sigma_noise : noise standard deviation
    g : activation function
    """
    # data
    x = iid_normal(dim=N, sample_shape=(P,), mu=mu_x, sigma=sigma_x) # (P, N)
    # weigths
    w = get_weights(N, out_dim, mu_w = mu_w, sigma_w = sigma_w) # (N, out_dim)
    # noise
    noise = iid_normal(dim=out_dim, sample_shape=(P,), mu=mu_noise, sigma=sigma_noise) # (P, out_dim)
    #noise = noise.squeeze() # (P, out_dim)
    y, _ = forward(w, x, N, g, denom=1.0) # (P, out_dim) 
    return x, y, noise, w

def get_data(seed, train_size, val_size, N, M, out_dim = 1, 
            k = None, singular_val=1.0,
            g = None,
            mu_x = 0.0, sigma_x = 1.0, # data
            mu_w = 0.0, sigma_w = 1.0, # w
            mu_noise = 0.0, sigma_noise = 1.0, # noise
            task = "regression"
             ):
    np.random.seed(seed)
    torch.manual_seed(seed)

    P = train_size+val_size
    x, y, noise_vec, w = teacher(N, P, out_dim = out_dim, g = g,
                                    mu_x = mu_x, sigma_x = sigma_x, # data
                                    mu_w = mu_w, sigma_w = sigma_w, # feature map
                                    mu_noise = mu_noise, sigma_noise = 1.0, # noise
                                    )

    F_matrix = torch.eye(N)
    if k is not None :
        F_matrix, _, _, _ = get_modulation_matrix_multi_singular(N, k, singular_val) # NxN
        F_matrix = torch.from_numpy(F_matrix).float()
    x = x @ F_matrix # P x N

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = val_size/P, random_state = seed)

    y_train = y_train + sigma_noise*noise_vec[:len(y_train)]
    # test data is noiseless
    y_test = y_test + 0*noise_vec[len(y_train):]
    
    if task == "classification" :
        y_train = binarize(y_train)
        y_test = binarize(y_test)

    return x_train, y_train, x_test, y_test, w, F_matrix

class DatasetWithIndexes(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)
    
def get_dataloader(train_size, val_size, N, 
    k=None, singular_val=1.0,
    out_dim = 1, g = None, 
    mu_x = 0.0, sigma_x = 1.0, # data
    mu_w = 0.0, sigma_w = 1.0, # w
    mu_noise = 0.0, sigma_noise = 1.0, # noise
    seed = 100, task = "regression", include_indexes = False, train_batch_size = None, val_batch_size = None, num_workers=0, return_just_set = True):
    """
    # train_size, val_size : number of training/validation examples
    # N: number of total dimensions
    # k: conditions numbers
    # noise: standard deviation of the noise added to the teacher output
    """
    x_train, y_train, x_test, y_test, w, F_matrix = get_data(seed, train_size, val_size, N, out_dim, 
                                    k=k, singular_val=singular_val,
                                    g=g,
                                    mu_x = mu_x, sigma_x = sigma_x, # data
                                    mu_w = mu_w, sigma_w = sigma_w, # w
                                    mu_noise = mu_noise, sigma_noise = sigma_noise, # noise
                                    task=task
                                    )
    data_infos = {}
    if "classification" in task :
        data_infos["pos_train"] = (y_train == 0).sum().item()
        data_infos["neg_train"] = (y_train == 1).sum().item()
        data_infos["pos_test"]  = (y_test == 0).sum().item()
        data_infos["neg_test"]  = (y_test == 1).sum().item()

    y_train, y_test = y_train.float(), y_test.float()
    #if task == "regression" : y_train, y_test = y_train.float(), y_test.float()
    #else : y_train, y_test = y_train.long(), y_test.long()

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

    return train_loader, val_loader, data_infos, w, F_matrix