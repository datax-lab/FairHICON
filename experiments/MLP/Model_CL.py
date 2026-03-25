import math
import copy
import torch
import numpy as np
import torch.nn as nn
from scipy import sparse
from Utils import fixed_s_mask
import torch.nn.functional as F


def init_parameters(self, init, weight, bias=None):
        if init == "he_normal":
            nn.init.kaiming_normal_(weight, mode='fan_in')
        elif init == "he_uniform":
            nn.init.kaiming_uniform_(weight, mode='fan_in')
        elif init == "xavier_normal":
            nn.init.xavier_normal_(weight)
        elif init == "xavier_uniform":
            nn.init.xavier_uniform_(weight)

        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
            
class Encoder(nn.Module):
    def __init__(self, net_hparams, sparse_indices):
        super(Encoder, self).__init__()        
        ### load sparse indices between features and groups
        self.indices = sparse_indices
        ### net_hparams
        ### 0-input_nodes, 1-hidden_nodes, 2-output_nodes, 3-initializer, 4-activation, 5-encoder_dropout 6-classifier_dropout        
        if net_hparams[4] == "sigmoid":
            self.layer1_activation = nn.Sigmoid()
            self.layer2_activation = nn.Sigmoid()
        elif net_hparams[4] == "tanh":
            self.layer1_activation = nn.Tanh()
            self.layer2_activation = nn.Tanh()
        elif net_hparams[4] == "relu":
            self.layer1_activation = nn.ReLU()
            self.layer2_activation = nn.ReLU()
        elif net_hparams[4] == "lkrelu":
            self.layer1_activation = nn.LeakyReLU()
            self.layer2_activation = nn.LeakyReLU()
        ### dropout
        if net_hparams[5] != 0.:
            self.on_Dropout = True
            self.dropout = nn.Dropout(net_hparams[5])
        else:
            self.on_Dropout = False
            
        self.layer1 = nn.Linear(net_hparams[0], net_hparams[1][0])
        init_parameters(net_hparams[3], self.layer1.weight.data, self.layer1.bias)
        self.layer2 = nn.Linear(net_hparams[1][0], net_hparams[1][1])
        init_parameters(net_hparams[3], self.layer2.weight.data, self.layer2.bias)
        self.layer3 = nn.Linear(net_hparams[1][1], net_hparams[1][2])
        init_parameters(net_hparams[3], self.layer3.weight.data, self.layer3.bias)
        ###########################################################################
        ### batch normalization & initialization
        ###########################################################################
        self.bn1 = nn.BatchNorm1d(net_hparams[1][0], eps=1e-3)
        self.bn2 = nn.BatchNorm1d(net_hparams[1][1], eps=1e-3)
        ###########################################################################        
    
    def forward(self, x):
        self.layer1.weight.data = fixed_s_mask(self.layer1.weight.data, self.indices)
        x = self.bn1(self.layer1_activation(self.layer1(x)))
        if self.on_Dropout:
            x = self.dropout(x)

        x = self.bn2(self.layer2_activation(self.layer2(x)))        
        if self.on_Dropout:
            x = self.dropout(x)
        
        x = self.layer3(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x


class Model(nn.Module):
    def __init__(self, net_hparams, sparse_indices):        
        super(Model, self).__init__()        
        ### net_hparams
        ### 0-input_nodes, 1-hidden_nodes, 2-output_nodes, 3-initializer, 4-activation, 5-encoder_dropout 6-classifier_dropout
        self.encoder_common = Encoder(net_hparams, sparse_indices)
        self.encoder_male = Encoder(net_hparams, sparse_indices)
        self.encoder_female = Encoder(net_hparams, sparse_indices)

    def forward(self, x):
        ### last column represents sex information
        ### 0 - female, 1 - male
        male_indices, female_indices = (x[:, -1] == 1), (x[:, -1] == 0)
        common_embeddings = self.encoder_common(x[:, :-1])
        male_embeddings = self.encoder_male(x[male_indices, :-1])
        female_embeddings = self.encoder_female(x[female_indices, :-1])
        
        # Combining male and female embeddings
        sex_specific_embeddings = torch.zeros_like(common_embeddings)
        sex_specific_embeddings[male_indices], sex_specific_embeddings[female_indices] = male_embeddings, female_embeddings
        
        return common_embeddings, sex_specific_embeddings