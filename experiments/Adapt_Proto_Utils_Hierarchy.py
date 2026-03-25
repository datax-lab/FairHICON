import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from scipy import sparse
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

import seaborn as sns
import umap

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

def plot_loss_plots(data, ver, date, num, experiment, train_loss_values, valid_loss_values):
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.plot(train_loss_values, color="#8B0000", label="train")
    plt.plot(valid_loss_values, color="#00008B", label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    # Add a common title for the figure
    fig.suptitle(f"{num}_learning_curves", fontsize=16)
    
    os.makedirs(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/{data}_Result/{ver}/Learning_Curve/", exist_ok=True)
    fig.savefig(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/{data}_Result/{ver}/Learning_Curve/[{date}_{num}]_[{experiment}]_Learning_Curve.png") 
    plt.close(fig)
    
def plot_all_loss_plots(data, ver, date, num, experiment, optim_hparams,
                        train_total_loss_values, valid_total_loss_values,
                        train_hcl_loss1_values, valid_hcl_loss1_values,
                        train_hcl_loss2_values, valid_hcl_loss2_values):
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))  # 2 rows, 2 columns
    
    # total loss plot
    axs[0, 0].plot(train_total_loss_values, color="#8B0000", label="train")
    axs[0, 0].plot(valid_total_loss_values, color="#00008B", label="valid")
    axs[0, 0].set_title("total loss")
    axs[0, 0].set_xlabel("epoch")
    axs[0, 0].set_ylabel("loss")
    axs[0, 0].legend()

    # CE loss plot
    # axs[0, 1].plot(train_ce_loss_values, color="#B22222", label="train")
    # axs[0, 1].plot(valid_ce_loss_values, color="#4169E1", label="valid")
    # axs[0, 1].set_title("CE loss")
    # axs[0, 1].set_xlabel("epoch")
    # axs[0, 1].set_ylabel("loss")
    # axs[0, 1].legend()
    
    # HCL1 loss plot
    axs[1, 0].plot(train_hcl_loss1_values, color="#DC143C", label="train")
    axs[1, 0].plot(valid_hcl_loss1_values, color="#1E90FF", label="valid")
    axs[1, 0].set_title("long term vs short term loss")
    axs[1, 0].set_xlabel("epoch")
    axs[1, 0].set_ylabel("loss")
    axs[1, 0].legend()

    # HCL2 loss plot
    axs[1, 1].plot(train_hcl_loss2_values, color="#FA8072", label="train")
    axs[1, 1].plot(valid_hcl_loss2_values, color="#87CEEB", label="valid")
    axs[1, 1].set_title("common vs male vs female loss")
    axs[1, 1].set_xlabel("epoch")
    axs[1, 1].set_ylabel("loss")
    axs[1, 1].legend()

    # Add a common title for the figure
    # fig.suptitle(f"{num}_learning_curves\nLR: {optim_hparams[1]}   LRD: {optim_hparams[10]}   LRDE: {optim_hparams[11]}", fontsize=16)
    # fig.suptitle(f"{num}_learning_curves\nLRF: {optim_hparams[2]}   LRD: {optim_hparams[10]}   LRDE: {optim_hparams[11]}", fontsize=16)
    fig.suptitle(f"{num}_learning_curves\nLRF: {optim_hparams[2]}   TAU1: {optim_hparams[5]}   TAU2: {optim_hparams[6]}", fontsize=16)
    # fig.suptitle(f"{num}_learning_curves", fontsize=16)
    # Adjust layout to prevent overlap
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/{data}_Result/{ver}/Learning_Curve/", exist_ok=True)
    fig.savefig(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/{data}_Result/{ver}/Learning_Curve/[{date}_{num}]_[{experiment}]_Learning_Curve.png")
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

def get_umap_embeddings(all_embeddings):
    fit = umap.UMAP(n_neighbors=max(2, int(0.07 * all_embeddings.shape[0])),    # balance local vs. global structure
                    min_dist=0.1,      # how tightly UMAP packs points together
                    n_components=2,    # output dimensionality
                    metric='euclidean',# distance metric
                    random_state=0)
    x_umap = fit.fit_transform(all_embeddings)
    return x_umap[:, 0], x_umap[:, 1]

def get_tsne_embeddings(embeddings_np):
    """Safely computes t-SNE embeddings, checking for sample size."""
    # --- CORRECTION: Add defensive check ---
    # t-SNE requires more samples than perplexity. A common rule of thumb is
    # at least 3 * perplexity. Here, we just check for a minimum of 2.
    if embeddings_np.shape[0] < 2:
        print(f"Warning: Skipping t-SNE. Not enough samples: {embeddings_np.shape[0]}")
        # Return empty arrays with the correct number of columns (2)
        return np.empty((embeddings_np.shape[0], 2))
    
    # Dynamically adjust perplexity
    perplexity = min(30, max(5, embeddings_np.shape[0] - 1))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    tsne_result = tsne.fit_transform(embeddings_np)
    return tsne_result

def get_tsne_embeddings(all_embeddings):    
    tsne = TSNE(n_components=2, random_state=0, max_iter=2000, n_iter_without_progress=1000, perplexity=max(2, int(0.15 * all_embeddings.shape[0])))
    # tsne = TSNE(n_components=2, random_state=0, perplexity=21)
    tsne_result = tsne.fit_transform(all_embeddings)
    tsne_x1 = tsne_result[:, 0]
    tsne_x2 = tsne_result[:, 1]
    return tsne_x1, tsne_x2

def get_common_groups(labels, sexs):
    common_group_labels = []
    common_plot_labels = []
    for lbl, sex in zip(labels, sexs):
        
        if lbl==1 and sex==1:
            common_group_labels.append("Common LTS")
            common_plot_labels.append("Common M LTS")
        
        elif lbl==1 and sex==0:
            common_group_labels.append("Common LTS")
            common_plot_labels.append("Common F LTS")
        
        elif lbl==0 and sex==1:
            common_group_labels.append("Common non_LTS")
            common_plot_labels.append("Common M non_LTS")
        
        elif lbl==0 and sex==0:
            common_group_labels.append("Common non_LTS")
            common_plot_labels.append("Common F non_LTS")
            
    return common_group_labels, common_plot_labels

def get_sex_specific_groups(labels, sexs):
    specific_labels = []    
    for lbl, sex in zip(labels, sexs):
        
        if lbl==1 and sex==1:
            specific_labels.append("Male LTS")
        
        elif lbl==1 and sex==0:
            specific_labels.append("Female LTS")
        
        elif lbl==0 and sex==1:
            specific_labels.append("Male non_LTS")
        
        elif lbl==0 and sex==0:
            specific_labels.append("Female non_LTS")
            
    return specific_labels

def plot_all_embeddings(data, ver, date, num, experiment, epoch, net, optim_hparams,
                        train_cmn_embeddings, train_spf_embeddings, train_labels, train_sexs,
                        valid_cmn_embeddings, valid_spf_embeddings, valid_labels, valid_sexs,
                        test_cmn_embeddings, test_spf_embeddings, test_labels, test_sexs):
    
    colors = [x for x in sns.color_palette("Paired")]
    
    # This creates a single, rich feature vector for each sample.
    train_embeddings = np.vstack([train_cmn_embeddings, train_spf_embeddings])
    valid_embeddings = np.vstack([valid_cmn_embeddings, valid_spf_embeddings])
    test_embeddings = np.vstack([test_cmn_embeddings, test_spf_embeddings])
    
    train_common_group_labels, train_common_plot_labels = get_common_groups(train_labels, train_sexs)
    valid_common_group_labels, valid_common_plot_labels = get_common_groups(valid_labels, valid_sexs)
    test_common_group_labels, test_common_plot_labels = get_common_groups(test_labels, test_sexs)
    train_specific_labels = get_sex_specific_groups(train_labels, train_sexs)
    valid_specific_labels = get_sex_specific_groups(valid_labels, valid_sexs)
    test_specific_labels = get_sex_specific_groups(test_labels, test_sexs)
    
    # Create Class Labels (LTS vs non-LTS) for all sets
    train_class_labels = np.vstack([train_labels, train_labels]).ravel()
    valid_class_labels = np.vstack([valid_labels, valid_labels]).ravel()
    test_class_labels = np.vstack([test_labels, test_labels]).ravel()
    
    # Create Hierarchical Group Labels (the 6 detailed groups) for all sets
    train_group_labels = train_common_group_labels + train_specific_labels
    valid_group_labels = valid_common_group_labels + valid_specific_labels
    test_group_labels = test_common_group_labels + test_specific_labels
    
    # Create Plot Labels for all sets
    train_plot_labels = train_common_plot_labels + train_specific_labels
    valid_plot_labels = valid_common_plot_labels + valid_specific_labels
    test_plot_labels = test_common_plot_labels + test_specific_labels
        
    # Score based on the primary class (LTS vs non-LTS)
    train_class_silhouette_score = silhouette_score(train_embeddings, train_class_labels)
    valid_class_silhouette_score = silhouette_score(valid_embeddings, valid_class_labels)
    test_class_silhouette_score = silhouette_score(test_embeddings, test_class_labels)

    # Score based on the detailed hierarchical groups
    train_group_silhouette_score = silhouette_score(train_embeddings, train_group_labels)
    valid_group_silhouette_score = silhouette_score(valid_embeddings, valid_group_labels)
    test_group_silhouette_score = silhouette_score(test_embeddings, test_group_labels)
    
    group_color_map = {
        "Common M LTS": colors[9],           # Dark Purple
        "Common M non_LTS": colors[8],       # Light Purple
        "Common F LTS": colors[9],           # Dark Purple
        "Common F non_LTS": colors[8],       # Light Purple

        "Male LTS": colors[1],               # Dark Blue
        "Male non_LTS": colors[0],           # Light Blue
        "Female LTS": colors[5],             # Dark Red
        "Female non_LTS": colors[4]          # Light Red
    }

    group_vs_marker = { "Common M LTS": '$\u2642$',
                        "Common M non_LTS": '$\u2642$',
                        "Common F LTS": '$\u2640$',           # Dark Purple
                        "Common F non_LTS": '$\u2640$',       # Light Purple

                        "Male LTS": '$\u2642$',               # Dark Blue
                        "Male non_LTS": '$\u2642$',           # Light Blue
                        "Female LTS": '$\u2640$',             # Dark Red
                        "Female non_LTS": '$\u2640$'
                      }
    
    tsne_train_x1, tsne_train_x2 = get_tsne_embeddings(train_embeddings)
    tsne_valid_x1, tsne_valid_x2 = get_tsne_embeddings(valid_embeddings)
    tsne_test_x1, tsne_test_x2 = get_tsne_embeddings(test_embeddings)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 9))

    # Plot Train Embeddings (Left Subplot)
    for group, color in group_color_map.items():
        #print("color: ", color)
        
        indices = [i for i, lbl in enumerate(train_plot_labels) if lbl == group]
        if indices:
            axs[0].scatter(
                [tsne_train_x1[i] for i in indices], [tsne_train_x2[i] for i in indices],
                c=[color], label=group, marker=group_vs_marker[group],  s=100
            )
            
    axs[0].set_title(f"Train Embeddings={train_class_silhouette_score:.4f}, {train_group_silhouette_score:.4f}")
    axs[0].legend()
    axs[0].set_xlabel("t-SNE x1")
    axs[0].set_ylabel("t-SNE x2")

    # Plot Valid Embeddings (Middle Subplot)
    for group, color in group_color_map.items():
        indices = [i for i, lbl in enumerate(valid_plot_labels) if lbl == group]
        if indices:
            axs[1].scatter(
                [tsne_valid_x1[i] for i in indices], [tsne_valid_x2[i] for i in indices],
                c=[color], label=group, marker=group_vs_marker[group], s=100
            )
            
    axs[1].set_title(f"Valid Embeddings={valid_class_silhouette_score:.4f}, {valid_group_silhouette_score:.4f}")
    axs[1].legend()
    axs[1].set_xlabel("t-SNE x1")
    axs[1].set_ylabel("t-SNE x2")

    # Plot Test Embeddings (Right Subplot)
    for group, color in group_color_map.items():
        #print("color: ", color)
        
        indices = [i for i, lbl in enumerate(test_plot_labels) if lbl == group]
        if indices:
            axs[2].scatter(
                [tsne_test_x1[i] for i in indices], [tsne_test_x2[i] for i in indices],
                c=[color], label=group, marker=group_vs_marker[group],  s=100
            )
            
    axs[2].set_title(f"Test Embeddings={test_class_silhouette_score:.4f}, {test_group_silhouette_score:.4f}")
    axs[2].legend()
    axs[2].set_xlabel("t-SNE x1")
    axs[2].set_ylabel("t-SNE x2")
    
    os.makedirs(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/{data}_Result/{ver}/Embeddings_Plot/", exist_ok=True)
    # Adjust layout and save the figure
    # fig.suptitle(f"All {net} Embeddings for {data}\nLR: {optim_hparams[1]}   LRD: {optim_hparams[10]}   LRDE: {optim_hparams[11]}", fontsize=16)
    # fig.suptitle(f"All {net} Embeddings for {data}\nLRF: {optim_hparams[2]}   LRD: {optim_hparams[10]}   LRDE: {optim_hparams[11]}", fontsize=16)
    fig.suptitle(f"All {net} Embeddings for {data}\nLRF: {optim_hparams[2]}   TAU1: {optim_hparams[5]}   TAU2: {optim_hparams[6]}", fontsize=16)
    # fig.suptitle(f"All {net} Embeddings for {data}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Space for the common title
    plt.savefig(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/{data}_Result/{ver}/Embeddings_Plot/[{date}_{num}]_[{experiment}]_{net}_Embeddings_Epoch{epoch}.png")
    plt.clf()
    plt.close(fig)
    
# --- Helper function to plot on a sphere to avoid repeating code ---
def plot_on_sphere(ax, x, y, z, groups, color_map, marker_map, show_legend=False, legend_anchor=None):
    """Plots data points on a 3D sphere on the given axes."""
    # Draw a wireframe sphere
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', linewidth=0.5, alpha=1.0)

    # Plot the scatter points
    for group_name, color in color_map.items():
        indices = (groups == group_name)
        if np.any(indices):
            depth_coords = y[indices]
            
            # 2. Map the depth coordinates [-1, 1] to an alpha range [min_alpha, 1].
            # Points with y=-1 (back) will be faint; points with y=1 (front) will be opaque.
            min_alpha = 0.01
            alpha_values = min_alpha + (depth_coords + 1) / 2 * (1 - min_alpha)
            
            # 3. Create an RGBA color array for this group.
            # Start with the group's base color.
            base_color_rgba = mcolors.to_rgba(color)
            # Create an array of this color for each point.
            rgba_colors = np.tile(base_color_rgba, (len(alpha_values), 1))
            # Replace the alpha channel (the 4th column) with our calculated depth values.
            rgba_colors[:, 3] = alpha_values
            
            min_size = 10
            max_size = 500
            size_values = min_size + (depth_coords + 1) / 2 * (max_size - min_size)
            
            ax.scatter(
                x[indices], y[indices], z[indices],
                c=rgba_colors,
                label=group_name,
                marker=marker_map[group_name], # Use .get for safety
                s=size_values,
                depthshade=False,
                edgecolor=None
            )
    
    # Set view and appearance
    # ax.set_title(title, fontsize=18, pad=20)
    ax.axis('off')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=45, azim=45)
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_position([0, 0, 1, 1])
    
    if show_legend and legend_anchor:
        ax.legend(loc='center left', bbox_to_anchor=legend_anchor, fontsize=20)
        
        