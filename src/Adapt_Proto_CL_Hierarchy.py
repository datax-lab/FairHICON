import pandas as pd
import random
import os
import numpy as np
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SupConLossGroupNorm(torch.nn.Module):
    """Supervised Contrastive Loss with Group-wise Normalization."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = temperature

    def forward(self, features, group_labels):
        device = features.device
        group_labels = group_labels.contiguous().view(-1, 1)
        
        pos_mask = torch.eq(group_labels, group_labels.T).float()
        unique_group_labels = torch.unique(group_labels)
        
        feature_dot = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(feature_dot, dim=1, keepdim=True)
        logits = feature_dot - logits_max.detach()
        
        logits_mask = torch.ones_like(pos_mask)
        logits_mask.fill_diagonal_(0)
        pos_mask = pos_mask * logits_mask
        
        # Denominator should include all pairs (positives and negatives)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-12)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        
        total_loss = 0.0        
        for group in unique_group_labels:
            mask_g = (group_labels == group).squeeze()  # (M,)
            if mask_g.sum() > 0:
                total_loss += loss[mask_g].mean()
        total_loss = total_loss / unique_group_labels.numel()

        return total_loss
    
class SupConAdapProtoHardLossGroupNorm(torch.nn.Module):
    """Adaptive Prototype-based Hard Negative Mining Loss."""
    def __init__(self, temperature=0.07, soft_neg_weight=3.0, hard_neg_weight=5.0):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = temperature
        self.soft_neg_weight = soft_neg_weight
        self.hard_neg_weight = hard_neg_weight

    def forward(self, features, group_labels, class_labels, sensitive_labels):
        device = features.device
        group_labels = group_labels.contiguous().view(-1, 1)
        class_labels = class_labels.contiguous().view(-1, 1)
        sensitive_labels = sensitive_labels.contiguous().view(-1, 1)
        
        unique_group_labels, inverse_indices = torch.unique(group_labels, return_inverse=True)
        inverse_indices = inverse_indices.squeeze()
        num_group_labels = unique_group_labels.numel()

        group_sum = torch.zeros((num_group_labels, features.shape[1]), device=device).index_add_(0, inverse_indices, features)
        group_count = torch.zeros(num_group_labels, device=device).index_add_(0, inverse_indices, torch.ones(features.shape[0], device=device))
        prototypes_tensor = F.normalize(group_sum / (group_count.unsqueeze(1) + 1e-12), p=2, dim=1)
        sample_prototypes = prototypes_tensor[inverse_indices]
        prototype_mat = torch.mm(features, sample_prototypes.T)

        pos_mask = torch.eq(group_labels, group_labels.T).float()
        logits_mask = torch.ones_like(pos_mask)
        logits_mask.fill_diagonal_(0)
        pos_mask = pos_mask * logits_mask

        class_mask = torch.eq(class_labels, class_labels.T).float()
        soft_neg_mask = class_mask * (~pos_mask.bool()) * logits_mask
        
        sensitive_mask = torch.eq(sensitive_labels, sensitive_labels.T).float()
        hard_neg_mask = (~class_mask.bool()) * sensitive_mask * logits_mask

        sim_mat = torch.mm(features, features.T)
        
        weight_matrix = torch.ones_like(sim_mat)
        soft_neg_weights = torch.exp(self.soft_neg_weight * (sim_mat - prototype_mat))
        hard_neg_weights = torch.exp(self.hard_neg_weight * (sim_mat - prototype_mat))
        
        weight_matrix = torch.where(soft_neg_mask.bool(), soft_neg_weights, weight_matrix)
        weight_matrix = torch.where(hard_neg_mask.bool(), hard_neg_weights, weight_matrix)
        weight_matrix = torch.where((sim_mat - prototype_mat) > 0, weight_matrix, 1.0)
        weight_matrix *= logits_mask

        feature_dot = sim_mat / self.temperature
        logits_max, _ = torch.max(feature_dot, dim=1, keepdim=True)
        logits = feature_dot - logits_max.detach()
        
        # --- CORRECTION ---
        # The denominator must include both positives and negatives.
        # `exp_logits_pos` for the positive pairs (unweighted).
        # `exp_logits_neg_weighted` for the negative pairs (weighted).
        exp_logits = torch.exp(logits)
        exp_logits_pos = exp_logits * pos_mask
        exp_logits_neg = exp_logits * (soft_neg_mask + hard_neg_mask)
        
        exp_logits_neg_weighted = weight_matrix * exp_logits_neg
        
        denominator = exp_logits_pos.sum(1, keepdim=True) + exp_logits_neg_weighted.sum(1, keepdim=True)
        log_prob = logits - torch.log(denominator + 1e-12)
        
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-12)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # Average first within each group, then across groups
        total_loss = 0.0
        for group in unique_group_labels:
            mask_g = (group_labels == group).squeeze()  # (M,)
            if mask_g.sum() > 0:
                total_loss += loss[mask_g].mean()
        total_loss = total_loss / num_group_labels

        return total_loss

def FairContrastiveLearning(common_embeddings, sex_specific_embeddings, mask_lts, mask_nlts, mask_ml, mask_fml, tau1=0.07, tau2=0.07, soft_neg=3., hard_neg=5., alpha=0.5):
    """Prepares data and computes the two hierarchical contrastive losses."""
    # device = common_embeddings.device
    device = common_embeddings[0].device # common_embeddings is now a tuple
    
    # Unpack the representation and projection
    common_rep, common_proj = common_embeddings
    specific_rep, specific_proj = sex_specific_embeddings
    
    groups_def_proj = {
        0: (common_proj, mask_nlts, 0, 0),
        1: (specific_proj, (mask_nlts) & (mask_ml), 0, 1),
        2: (specific_proj, (mask_nlts) & (mask_fml), 0, 2),
        3: (common_proj, mask_lts, 1, 0),
        4: (specific_proj, (mask_lts) & (mask_ml), 1, 1),
        5: (specific_proj, (mask_lts) & (mask_fml), 1, 2)
    }
    
    all_views_proj, group_labels, class_labels, sensitive_labels = [], [], [], []
    for group_id in range(len(groups_def_proj)):
        embeds, mask, class_id, sens_id = groups_def_proj[group_id]
        views = embeds[mask.bool()]
        if views.shape[0] > 0:
            all_views_proj.append(views)
            group_labels.extend([group_id] * views.shape[0])
            class_labels.extend([class_id] * views.shape[0])
            sensitive_labels.extend([sens_id] * views.shape[0])

    if not all_views_proj:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    all_views_proj_tensor = torch.cat(all_views_proj, dim=0)
    group_labels = torch.tensor(group_labels, dtype=torch.long, device=device)
    class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)
    sensitive_labels = torch.tensor(sensitive_labels, dtype=torch.long, device=device)
    
    criterion1 = SupConLossGroupNorm(temperature=tau1)
    criterion2 = SupConAdapProtoHardLossGroupNorm(
        temperature=tau2,
        soft_neg_weight=soft_neg,
        hard_neg_weight=hard_neg
    )
    
    hcl_loss1_proj = criterion1(all_views_proj_tensor, class_labels)
    hcl_loss2_proj = criterion2(all_views_proj_tensor, group_labels, class_labels, sensitive_labels)

    groups_def_rep = {
        0: (common_rep, mask_nlts, 0, 0),
        1: (specific_rep, (mask_nlts) & (mask_ml), 0, 1),
        2: (specific_rep, (mask_nlts) & (mask_fml), 0, 2),
        3: (common_rep, mask_lts, 1, 0),
        4: (specific_rep, (mask_lts) & (mask_ml), 1, 1),
        5: (specific_rep, (mask_lts) & (mask_fml), 1, 2)
    }
    
    all_views_rep = []
    # We can reuse the labels since the samples are the same
    for group_id in range(len(groups_def_rep)):
        embeds, mask, _, _ = groups_def_rep[group_id]
        views = embeds[mask.bool()]
        if views.shape[0] > 0:
            all_views_rep.append(views)

    all_views_rep_tensor = torch.cat(all_views_rep, dim=0)    
    # --- CRITICAL FIX FOR NUMERICAL STABILITY ---
    # L2-normalize the representation vectors *only for this loss calculation*.
    # This prevents the dot product from exploding while preserving the
    # un-normalized z_rep for downstream tasks and plotting.
    all_views_rep_tensor = F.normalize(all_views_rep_tensor, p=2, dim=1)
    
    # Note: We do NOT L2-normalize the representation before the loss
    hcl_loss1_rep = criterion1(all_views_rep_tensor, class_labels)
    hcl_loss2_rep = criterion2(all_views_rep_tensor, group_labels, class_labels, sensitive_labels)
    
    # --- Combine the losses ---
    # `alpha` is a new hyperparameter to weigh the representation loss
    # alpha = hparams.get('alpha', 0.5)
    # print(hcl_loss1_proj, hcl_loss1_rep)
    # print(hcl_loss2_proj, hcl_loss2_rep)
    final_hcl_loss1 = hcl_loss1_proj + alpha * hcl_loss1_rep
    final_hcl_loss2 = hcl_loss2_proj + alpha * hcl_loss2_rep
    
    return final_hcl_loss1, final_hcl_loss2
