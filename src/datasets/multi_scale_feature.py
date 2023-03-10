from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch

import numpy as np
from scipy.stats import ortho_group

def d_div_m_uniform(d, m) :
    """
    We want to share d items between m subjects.
    We give 1 to the 1st, 1 to the 2nd, ...., 1 to the m-th. 
    Then we repeat the same thing. 
    We stop when there are no more items to distribute.
    """
    assert d >= m
    p = np.zeros((m,), dtype=int) # subjects
    for i in range(d) : p[i%m]+=1
    return p
    
def get_S(d, k):
    """
    Rectangular diagonal matrix of the modulation matrix.
    In this implementation we choose a Diagonal matrix.

    The 1st p elements of the diagonal matrix are equal to 1/k_1 = 1
    ... 2nd .............................................. 1/(k_1*k_2)
    ... 3nd ............................................... 1/(k_1*k_2*k_3)
    """
    try : k.insert(0, 1)
    except AttributeError : k = [1, k] # if was int, not array
    k = np.array(k)
    m = len(k) 
    p = d_div_m_uniform(d, m)
    S = np.eye(d)
    for i in range(m) :
        a, b = sum(p[:i]), sum(p[:i+1])
        S[a:b, a:b] *= 1/k[:i+1].prod()
    return S

def get_modulation_matrix_multi_singular(d, k):
    """
    d : int (size of the modulation matrix dxd)
    k : list of conditions numbers ([k_1, ..., k_m])
    """
    U = ortho_group.rvs(d)
    VT = ortho_group.rvs(d)
    S = get_S(d, k)
    F = np.dot(U, np.dot(S, VT))
    # F = S
    return F, U, VT, S

def get_modulation_matrix(d, p, k):
    F, U, VT, S = get_modulation_matrix_multi_singular(d, [k/2, k, k*10])
    # U = ortho_group.rvs(d)
    # VT = ortho_group.rvs(d)
    # S = np.eye(d)
    # S[:p, :p] *= 1
    # S[p:, p:] *= 1 / k
    # F = np.dot(U, np.dot(S, VT))
    # # F = S
    return F

# Implements the teacher and generates the data
def get_data(seed, train_size, val_size, d, k, noise):
    np.random.seed(seed)
    Z = np.random.randn(train_size, d) / np.sqrt(d)
    Z_test = np.random.randn(val_size, d) / np.sqrt(d)

    # teacher
    w = np.random.randn(d, 1)
    y = np.dot(Z, w)
    y = y + noise * np.random.randn(*y.shape)
    # test data is noiseless
    y_test = np.dot(Z_test, w)

    # the modulation matrix that controls students access to the data
    #F = get_modulation_matrix(d, p, k)
    F, _, _, _ = get_modulation_matrix_multi_singular(d, k)

    # X = F^T Z
    X = np.dot(Z, F)
    X_test = np.dot(Z_test, F)

    return X, y, X_test, y_test, F, w

class DatasetWithIndexes(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)
    
def get_dataloader(train_size, val_size, d, k, noise = 0.0, seed = 100, task = "regression", include_indexes = False, train_batch_size = None, val_batch_size = None, num_workers=0, return_just_set = True):
    """
    # train_size, val_size : number of training/validation examples
    # d: number of total dimensions
    # k: conditions numbers
    # noise: standard deviation of the noise added to the teacher output
    """
    X, y, X_test, y_test, _, _ = get_data(seed, train_size, val_size, d, k, noise)

    y, y_test = torch.from_numpy(y), torch.from_numpy(y_test)
    if task == "regression" : y, y_test = y.float(), y_test.float()
    else : y, y_test = y.long(), y_test.long()

    train_set = TensorDataset(torch.from_numpy(X).float(), y)
    val_set = TensorDataset(torch.from_numpy(X_test).float(), y_test)

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
