import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ------------------------------------------------------------
# 1) Parameter Initialization Helper
# ------------------------------------------------------------
def init_parameters(m, init="he_normal"):
    """
    Initialize parameters of a Linear layer using the specified init method.
    Supported inits: "he_normal", "he_uniform", "xavier_normal", "xavier_uniform".
    Also initializes bias uniformly in [-1/sqrt(fan_in), +1/sqrt(fan_in)].
    """
    if isinstance(m, nn.Linear):
        if init == "he_normal":
            nn.init.kaiming_normal_(m.weight, mode="fan_in")
        elif init == "he_uniform":
            nn.init.kaiming_uniform_(m.weight, mode="fan_in")
        elif init == "xavier_normal":
            nn.init.xavier_normal_(m.weight)
        elif init == "xavier_uniform":
            nn.init.xavier_uniform_(m.weight)
        # Bias initialization (uniform in [-bound, bound])
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

# ------------------------------------------------------------
# 2) “Sparse” Encoder
# ------------------------------------------------------------
class GeneExpressionEncoder(nn.Module):
    def __init__(self, net_hparams, sparse_indices):
        """
        net_hparams: list or tuple containing:
          0: input_dim (number of genes)
          1: hidden_dims [pathway_dim, hidden_dim]
          2: (not used here)
          3: initializer string ("he_normal", etc.)
          4: activation string ("sigmoid","tanh","relu","lkrelu")
          5: encoder_dropout (float)
        sparse_indices: a mask (same shape as layer1.weight) to zero out certain weights on each forward.
        """
        super().__init__()
        ### Load sparse idxs between genes and pathways
        self.indices = sparse_indices

        # Activation functions
        act = net_hparams[4]
        if act == "sigmoid":
            self.act1 = nn.Sigmoid();  self.act2 = nn.Sigmoid()
        elif act == "tanh":
            self.act1 = nn.Tanh();     self.act2 = nn.Tanh()
        elif act == "relu":
            self.act1 = nn.ReLU();     self.act2 = nn.ReLU()
        elif act == "lkrelu":
            self.act1 = nn.LeakyReLU(); self.act2 = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation '{act}'")

        # Dropout
        self.on_Dropout = (net_hparams[5] != 0.0)
        if self.on_Dropout:
            self.dropout = nn.Dropout(net_hparams[5])

        # Gene → Pathway
        input_dim = net_hparams[0]
        pathway_dim = net_hparams[1][0]
        self.layer1 = nn.Linear(input_dim, pathway_dim)
        init_parameters(self.layer1, init=net_hparams[3])
        self.bn1 = nn.BatchNorm1d(pathway_dim, momentum=0.01)

        # Pathway → Hidden
        hidden_dim = net_hparams[1][1]
        self.layer2 = nn.Linear(pathway_dim, hidden_dim)
        init_parameters(self.layer2, init=net_hparams[3])
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.01)
        
    def forward(self, x):
        # Apply on‐the‐fly mask to layer1 weights
        self.layer1.weight.data = fixed_s_mask(self.layer1.weight.data, self.indices)

        # Layer1 → BN → Activation → (Optionally Dropout)
        x = self.bn1(self.layer1(x))
        x = self.act1(x)
        if self.on_Dropout:
            x = self.dropout(x)

        # Layer2 → BN → Activation → (Optionally Dropout)
        x = self.bn2(self.layer2(x))
        x = self.act2(x)
        if self.on_Dropout:
            x = self.dropout(x)
            
        return x            

# ------------------------------------------------------------
# 3) CLNet = Encoder + Projection Head
# ------------------------------------------------------------
class GeneExpressionBackbone(nn.Module):
    """
    Wraps one Encoder + a small projection head.
    net_hparams: same as for Encoder, but net_hparams[2] = output_dim of projection head.
    """
    def __init__(self, net_hparams, sparse_indices):
        super().__init__()

        act = net_hparams[4]
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "lkrelu":
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation '{act}'")

        self.encoder = GeneExpressionEncoder(net_hparams, sparse_indices)

        # Projection head: hidden_dim → representation_dim → prjection_dim
        hidden_dim = net_hparams[1][1]
        proj_dim = net_hparams[1][2]
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            self.act,
            nn.Dropout(net_hparams[5]),
            nn.Linear(hidden_dim, proj_dim)
        )
        # This minimalist head has the least possible capacity, forcing the
        # encoder to do almost all of the work to separate the classes.
        # self.head = nn.Linear(hidden_dim, proj_dim)

        # Initialize all parameters
        self.apply(lambda m: init_parameters(m, init=net_hparams[3]))

    def forward(self, x):
        # 1) Encoder representation (z_rep)
        z_rep = self.encoder(x) # shape: (batch_size, hidden_dim)

        # 2) Projection → L2‐normalize (z_proj)
        # z_proj = self.head(z_rep)
        z_proj = F.normalize(self.head(z_rep), p=2, dim=1) # shape: (batch_size, proj_dim)
        return z_rep, z_proj

# ------------------------------------------------------------
# 4) CLModule: three CLNets (common/male/female)
# ------------------------------------------------------------
class CLModule(nn.Module):
    """
    Combines:
      - encoder_common
      - encoder_male
      - encoder_female
    into a single module.  Forward(x) returns:
      (z_rep_common, z_proj_common), (z_rep_specific, z_proj_specific)
    where the last column of x is “sex” (0 or 1).
    """
    def __init__(self, net_hparams, sparse_indices):
        super().__init__()
        self.encoder_common = GeneExpressionBackbone(net_hparams, sparse_indices)
        self.encoder_male = GeneExpressionBackbone(net_hparams, sparse_indices)
        self.encoder_female = GeneExpressionBackbone(net_hparams, sparse_indices)

    def forward(self, x):
        # x shape: (N, input_dim+1), last column = sex ∈ {0,1}
        sex = x[:, -1].long()     # shape: (N,)
        feats = x[:, :-1]         # shape: (N, input_dim)

        # 1) Common embeddings (for all samples)
        z_rep_common, z_proj_common = self.encoder_common(feats)

        # 2) Sex-specific embeddings
        male_idxs = (sex == 1).nonzero(as_tuple=True)[0]
        female_idxs = (sex == 0).nonzero(as_tuple=True)[0]

        N = feats.size(0)
        repr_dim = z_rep_common.size(1)
        proj_dim = z_proj_common.size(1)

        z_rep_specific = torch.zeros((N, repr_dim), device=feats.device)
        z_proj_specific = torch.zeros((N, proj_dim), device=feats.device)

        if male_idxs.numel() > 0:
            x_m = feats[male_idxs]
            zr_m, zp_m = self.encoder_male(x_m)
            z_rep_specific[male_idxs]  = zr_m
            z_proj_specific[male_idxs] = zp_m

        if female_idxs.numel() > 0:
            x_f = feats[female_idxs]
            zr_f, zp_f = self.encoder_female(x_f)
            z_rep_specific[female_idxs]  = zr_f
            z_proj_specific[female_idxs] = zp_f

        return (z_rep_common, z_proj_common), (z_rep_specific, z_proj_specific)
