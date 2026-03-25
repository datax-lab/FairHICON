import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import sparse
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score


def load_sparse_indices(path):
    coo = sparse.load_npz(path)
    indices = np.vstack((coo.row, coo.col))
    return indices

def fixed_s_mask(w, idx):
    '''
    Input: 
        w: weight matrix.
        idx: the indices of having values (or connections).
    Output:
        returns the weight matrix that has been forced the connections.
    '''
    sp_w = torch.sparse_coo_tensor(idx, w[idx[0], idx[1]], w.size())
    
    return sp_w.to_dense()


def plot_loss_plots(data, ver, model_num, date, num, experiment,
                    train_loss_values, valid_loss_values):
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.plot(train_loss_values, color="#8B0000", label="train")
    plt.plot(valid_loss_values, color="#00008B", label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title(f"[{num}_{experiment}]_learning_curves")
    os.makedirs(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/MLP/{data}/{ver}/Learning_Curve/", exist_ok=True)
    fig.savefig(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/MLP/{data}/{ver}/Learning_Curve/[{date}_{num}]_[{experiment}]_Learning_Curve_{model_num}.png")
    plt.clf()
    plt.close(fig)
    

def auc(y_true, y_pred, sample_weight = None):
    ###if gpu is being used, transferring back to cpu
    if torch.cuda.is_available():
        y_true = y_true.cpu().detach()
        y_pred = y_pred.cpu().detach()
        
    if sample_weight is None:
        auc = roc_auc_score(y_true.numpy().ravel(), y_pred.numpy().ravel())
    else:
        sample_weight = sample_weight.cpu().detach()
        auc = roc_auc_score(y_true.numpy().ravel(), y_pred.numpy().ravel(), sample_weight = sample_weight)
    
    return auc
