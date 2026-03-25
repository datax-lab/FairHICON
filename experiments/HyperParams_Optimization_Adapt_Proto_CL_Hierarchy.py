import pandas as pd
import numpy as np
import random
import os

from ray import tune
from ray import train
from ray.air import session
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import ray.cloudpickle as pickle

from Adapt_Proto_Utils_Hierarchy import *
from Model import *
from Adapt_Proto_CL_Hierarchy import *

from tqdm import tqdm
from sklearn.utils import class_weight
from datetime import datetime
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str, help = "Types of data", required=True)
parser.add_argument("--ver", type = str, help = "The version number", required=True)
parser.add_argument("--dim", type = int, help = "Projection Dimensions", required=True)
parser.add_argument("--act", type = str, help = "Activation", required=True)
args = parser.parse_args()

date = datetime.today().strftime('%m%d')
data = args.data
ver = args.ver

def read_cancerData(experiment, cancer):
    trainData = pd.read_csv(f"/home/koe3/Bioinformatics/Data/{cancer}/0121_{cancer}_Normed_Train_Data_{experiment}.csv")
    trainLabel = pd.read_csv(f"/home/koe3/Bioinformatics/Data/{cancer}/0121_{cancer}_Normed_Train_Label_{experiment}.csv")
    validData = pd.read_csv(f"/home/koe3/Bioinformatics/Data/{cancer}/0121_{cancer}_Normed_Valid_Data_{experiment}.csv")
    validLabel = pd.read_csv(f"/home/koe3/Bioinformatics/Data/{cancer}/0121_{cancer}_Normed_Valid_Label_{experiment}.csv")    
    testData = pd.read_csv(f"/home/koe3/Bioinformatics/Data/{cancer}/0121_{cancer}_Normed_Test_Data_{experiment}.csv")
    testLabel = pd.read_csv(f"/home/koe3/Bioinformatics/Data/{cancer}/0121_{cancer}_Normed_Test_Label_{experiment}.csv")

    return trainData, trainLabel, validData, validLabel, testData, testLabel

def read_asthmaData(experiment, asthma):    
    trainData = pd.read_csv(f"/home/koe3/Bioinformatics/Data/Asthma/{asthma}/{asthma}_Normed_Train_Data_{experiment}.csv")
    trainLabel = pd.read_csv(f"/home/koe3/Bioinformatics/Data/Asthma/{asthma}/{asthma}_Normed_Train_Label_{experiment}.csv")
    validData = pd.read_csv(f"/home/koe3/Bioinformatics/Data/Asthma/{asthma}/{asthma}_Normed_Valid_Data_{experiment}.csv")
    validLabel = pd.read_csv(f"/home/koe3/Bioinformatics/Data/Asthma/{asthma}/{asthma}_Normed_Valid_Label_{experiment}.csv")
    testData = pd.read_csv(f"/home/koe3/Bioinformatics/Data/Asthma/{asthma}/{asthma}_Normed_Test_Data_{experiment}.csv")
    testLabel = pd.read_csv(f"/home/koe3/Bioinformatics/Data/Asthma/{asthma}/{asthma}_Normed_Test_Label_{experiment}.csv")

    return trainData, trainLabel, validData, validLabel, testData, testLabel

def get_num_nodes(data_name):
    """Returns the number of input features (genes) for a given dataset."""
    node_map = {
        "LIHC": 4781, "STAD": 4790, "LUAD": 4786,
        "LUSC": 4787, "LGG": 4789, "GSE240567": 4500
    }
    if data_name not in node_map:
        raise ValueError(f"Unknown dataset name for get_num_nodes: {data_name}")
        
    return node_map[data_name]

def fit(config):
    """
        Terminology:
            lts: long term survival
            nlts: non-long term survival
            cmn: common
            ml: male
            fml: female
            es: early stopping
    """
    # Use CUDA when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    experiment = 1
    num = str(session.get_trial_id())
    # print(num)
    ### MODEL PARAMETERS LOG
    In_Nodes = get_num_nodes(data)
    if data == "GSE240567":
        sparse_indices = load_sparse_indices(f"/home/koe3/Bioinformatics/Data/Asthma/{data}/{data}_Gene_Pathway_Mask.npz")
        trainData, trainLabel, validData, validLabel, testData, testLabel = read_asthmaData(experiment, data)
    else:
        sparse_indices = load_sparse_indices(f"/home/koe3/Bioinformatics/Data/TCGA_{data}_Pathway_Mask.npz")
        trainData, trainLabel, validData, validLabel, testData, testLabel = read_cancerData(experiment, data)
    
    ### 0-input_nodes, 1-hidden_nodes, 2-output_nodes, 3-initializer, 4-activation, 5-encoder_dropout
    net_hparams = [In_Nodes, [395, 128, args.dim], 1, config['INIT'], args.act, config['ENC_DR']]
    ### 0-optimizer, 1-lr, 2-cos_restart, 3-cos_fact, 4-weight_decay, 5-tau1, 6-tau2, 7-lambda, 8-sg, 9-hn, 10-alpha
    optim_hparams = [config['OPT'], config['LR'], 250, 1, config['WD'], config['TAU1'], config['TAU2'], config['LAMBD'], config['SOFT_NEG'], config['HARD_NEG'], config['ALPHA']]
    ### 0-max_epoch, 1-min_epoch, 2-earlystop_patience, 3-phase2_start_epoch, 4-warmup_epoch
    experim_hparms = [3000, 2000, 1000, 501, 500]

    # Get Train Data, Train Label, Valid Data and Valid Label
    train_x = torch.from_numpy(trainData.values).float().to(device, non_blocking=True)
    train_y = torch.from_numpy(trainLabel.values).float().to(device, non_blocking=True)
    valid_x = torch.from_numpy(validData.values).float().to(device, non_blocking=True)
    valid_y = torch.from_numpy(validLabel.values).float().to(device, non_blocking=True)
    test_x = torch.from_numpy(testData.values).float().to(device, non_blocking=True)
    test_y = torch.from_numpy(testLabel.values).float().to(device, non_blocking=True)
        
    train_mask_lts, train_mask_nlts = (train_y == 1).squeeze(), (train_y == 0).squeeze()
    train_mask_ml, train_mask_fml = (train_x[:, -1] == 1), (train_x[:, -1] == 0)
    
    valid_mask_lts, valid_mask_nlts = (valid_y == 1).squeeze(), (valid_y == 0).squeeze()
    valid_mask_ml, valid_mask_fml = (valid_x[:, -1] == 1), (valid_x[:, -1] == 0)

    net = CLModule(net_hparams, sparse_indices).to(device)
    ### Optimizer Setting
    ### 0-optimizer, 1-lr, 2-lr_factor, 3-lr_patience, 4-weight_decay, 5-lr_decay_rate, 6-lr_decay_epochs
    if optim_hparams[0] == "Adam":
        opt = optim.Adam(net.parameters(), lr=optim_hparams[1], betas=(0.99, 0.999), weight_decay=optim_hparams[4], amsgrad=True)
    elif optim_hparams[0] == "AdamW":
        opt = optim.AdamW(net.parameters(), lr=optim_hparams[1], betas=(0.99, 0.999), weight_decay=optim_hparams[4], amsgrad=True)
    ### Learning scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=optim_hparams[2], T_mult=optim_hparams[3], eta_min=1e-7)
    
    train_total_loss_values, train_hcl_loss1_values, train_hcl_loss2_values = [], [], []
    valid_total_loss_values, valid_hcl_loss1_values, valid_hcl_loss2_values = [], [], []
    
    best_silhouette_score = -1 # Silhouette score ranges from -1 to 1
    best_state = None
    opt_epoch = 1
    earlystopping_patience = experim_hparms[2]
    
    phase2_start_epoch = experim_hparms[3]
    final_lambd = optim_hparams[7]
    warmup_epochs = experim_hparms[4]
    
    for epoch in range(1, experim_hparms[0] + 1):          
        net.train()
        train_cmn_embeddings, train_spf_embeddings = net(train_x)
        train_hcl_loss1, train_hcl_loss2 = FairContrastiveLearning(train_cmn_embeddings, train_spf_embeddings,
                                                                   train_mask_lts, train_mask_nlts, train_mask_ml, train_mask_fml,
                                                                   optim_hparams[5], optim_hparams[6], optim_hparams[8], optim_hparams[9], optim_hparams[10])

        current_lambd = 0.0
        if epoch >= phase2_start_epoch:
            # Calculate how many epochs we are into the warm-up phase
            epochs_into_phase2 = epoch - phase2_start_epoch
            if epochs_into_phase2 < warmup_epochs:
                # Linearly increase lambda during the warm-up period
                current_lambd = final_lambd * (epochs_into_phase2 / warmup_epochs)
            else:
                # After warm-up, use the final lambda value
                current_lambd = final_lambd
        
        train_total_loss = train_hcl_loss1 + (current_lambd * train_hcl_loss2)
        
        opt.zero_grad()
        train_total_loss.backward()
        ### force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
        if net.encoder_common.encoder.layer1.weight.grad is not None:
            net.encoder_common.encoder.layer1.weight.grad = fixed_s_mask(net.encoder_common.encoder.layer1.weight.grad, sparse_indices)        
        if net.encoder_male.encoder.layer1.weight.grad is not None:
            net.encoder_male.encoder.layer1.weight.grad = fixed_s_mask(net.encoder_male.encoder.layer1.weight.grad, sparse_indices)        
        if net.encoder_female.encoder.layer1.weight.grad is not None:
            net.encoder_female.encoder.layer1.weight.grad = fixed_s_mask(net.encoder_female.encoder.layer1.weight.grad, sparse_indices)        
        opt.step()
        scheduler.step()
        
        net.eval()
        with torch.no_grad():
            valid_cmn_embeddings, valid_spf_embeddings = net(valid_x)
            valid_hcl_loss1, valid_hcl_loss2 = FairContrastiveLearning(valid_cmn_embeddings, valid_spf_embeddings,
                                                                       valid_mask_lts, valid_mask_nlts, valid_mask_ml, valid_mask_fml,
                                                                       optim_hparams[5], optim_hparams[6], optim_hparams[8], optim_hparams[9], optim_hparams[10])
            
            valid_total_loss = (valid_hcl_loss1 + (final_lambd * valid_hcl_loss2)).item()
            train_total_loss = train_total_loss.item()
            
            train_hcl_loss1_values.append(train_hcl_loss1.item())
            valid_hcl_loss1_values.append(valid_hcl_loss1.item())

            train_hcl_loss2_values.append((final_lambd * train_hcl_loss2).item())
            valid_hcl_loss2_values.append((final_lambd * valid_hcl_loss2).item())

            train_total_loss_values.append(train_total_loss)
            valid_total_loss_values.append(valid_total_loss)
            
            valid_rep_embeddings = torch.cat((valid_cmn_embeddings[0], valid_spf_embeddings[0]), dim=0)
            # Create the labels for the silhouette score
            valid_class_labels = torch.cat((valid_y, valid_y), dim=0).squeeze()
            # print(valid_rep_embeddings.shape)
            # print(valid_class_labels.shape)
            # Calculate silhouette score
            # Ensure there are at least 2 clusters to score
            if len(torch.unique(valid_class_labels)) > 1:
                current_silhouette = silhouette_score(valid_rep_embeddings.cpu().numpy(), valid_class_labels.cpu().numpy())
            else:
                current_silhouette = -1 # Cannot compute score
        # print(current_silhouette)
        if current_silhouette > best_silhouette_score:
            best_silhouette_score = current_silhouette
            best_state = copy.deepcopy(net.state_dict())
            opt_epoch = epoch
            earlystopping_patience = experim_hparms[2] # Reset patience
            # print(f"\nNew best silhouette score: {best_silhouette_score:.4f} at epoch {epoch}")
        else:
            earlystopping_patience -= 1
            if (experim_hparms[1] <= epoch) and (earlystopping_patience == 0):
                print(f"[{num}] Early stopping in epoch [{epoch}]")
                break

        if epoch % 200 == 0:
            plot_all_loss_plots(data, ver, date, num, experiment, optim_hparams,
                                train_total_loss_values, valid_total_loss_values,
                                train_hcl_loss1_values, valid_hcl_loss1_values,
                                train_hcl_loss2_values, valid_hcl_loss2_values)
    
    plot_all_loss_plots(data, ver, date, num, experiment, optim_hparams, train_total_loss_values, valid_total_loss_values,
                        train_hcl_loss1_values, valid_hcl_loss1_values, train_hcl_loss2_values, valid_hcl_loss2_values)
    
    # Instantiate a new model (same architecture)
    opt_net = CLModule(net_hparams, sparse_indices).to(device)
    # Load the best weights you stashed in best_state
    opt_net.load_state_dict(best_state)
    # Put into eval mode
    opt_net.eval()
    with torch.no_grad():
        opt_train_cmn_embeddings, opt_train_spf_embeddings = opt_net(train_x)
        opt_valid_cmn_embeddings, opt_valid_spf_embeddings = opt_net(valid_x)
        opt_test_cmn_embeddings, opt_test_spf_embeddings = opt_net(test_x)
        
        plot_all_embeddings(data, ver, date, num, experiment, opt_epoch, "OPT_REPR", optim_hparams,
                            opt_train_cmn_embeddings[0].cpu().detach().numpy(), opt_train_spf_embeddings[0].cpu().detach().numpy(),
                            train_y.cpu().detach().numpy(), train_x[:, -1].cpu().detach().numpy(),
                            opt_valid_cmn_embeddings[0].cpu().detach().numpy(), opt_valid_spf_embeddings[0].cpu().detach().numpy(),
                            valid_y.cpu().detach().numpy(), valid_x[:, -1].cpu().detach().numpy(),   
                            opt_test_cmn_embeddings[0].cpu().detach().numpy(), opt_test_spf_embeddings[0].cpu().detach().numpy(),
                            test_y.cpu().detach().numpy(), test_x[:, -1].cpu().detach().numpy())
        plot_all_embeddings(data, ver, date, num, experiment, opt_epoch, "OPT_PROJ", optim_hparams,
                            opt_train_cmn_embeddings[1].cpu().detach().numpy(), opt_train_spf_embeddings[1].cpu().detach().numpy(),
                            train_y.cpu().detach().numpy(), train_x[:, -1].cpu().detach().numpy(),
                            opt_valid_cmn_embeddings[1].cpu().detach().numpy(), opt_valid_spf_embeddings[1].cpu().detach().numpy(),
                            valid_y.cpu().detach().numpy(), valid_x[:, -1].cpu().detach().numpy(),   
                            opt_test_cmn_embeddings[1].cpu().detach().numpy(), opt_test_spf_embeddings[1].cpu().detach().numpy(),
                            test_y.cpu().detach().numpy(), test_x[:, -1].cpu().detach().numpy())

    net.eval()
    with torch.no_grad():
        final_train_cmn_embeddings, final_train_spf_embeddings = net(train_x)
        final_valid_cmn_embeddings, final_valid_spf_embeddings = net(valid_x)
        final_test_cmn_embeddings, final_test_spf_embeddings = net(test_x)

        plot_all_embeddings(data, ver, date, num, experiment, epoch, "FINAL_REPR", optim_hparams,
                            final_train_cmn_embeddings[0].cpu().detach().numpy(), final_train_spf_embeddings[0].cpu().detach().numpy(),
                            train_y.cpu().detach().numpy(), train_x[:, -1].cpu().detach().numpy(),
                            final_valid_cmn_embeddings[0].cpu().detach().numpy(), final_valid_spf_embeddings[0].cpu().detach().numpy(),
                            valid_y.cpu().detach().numpy(), valid_x[:, -1].cpu().detach().numpy(),
                            final_test_cmn_embeddings[0].cpu().detach().numpy(), final_test_spf_embeddings[0].cpu().detach().numpy(),
                            test_y.cpu().detach().numpy(), test_x[:, -1].cpu().detach().numpy())
        plot_all_embeddings(data, ver, date, num, experiment, epoch, "FINAL_PROJ", optim_hparams,
                            final_train_cmn_embeddings[1].cpu().detach().numpy(), final_train_spf_embeddings[1].cpu().detach().numpy(),
                            train_y.cpu().detach().numpy(), train_x[:, -1].cpu().detach().numpy(),
                            final_valid_cmn_embeddings[1].cpu().detach().numpy(), final_valid_spf_embeddings[1].cpu().detach().numpy(),
                            valid_y.cpu().detach().numpy(), valid_x[:, -1].cpu().detach().numpy(),
                            final_test_cmn_embeddings[1].cpu().detach().numpy(), final_test_spf_embeddings[1].cpu().detach().numpy(),
                            test_y.cpu().detach().numpy(), test_x[:, -1].cpu().detach().numpy())
            
    train.report({"Training total loss": train_total_loss, "Training hcl loss1": train_hcl_loss1.item(), "Training hcl loss2": train_hcl_loss2.item(),
                  "Validation total loss": valid_total_loss, "Validation hcl loss1": valid_hcl_loss1.item(), "Validation hcl loss2": valid_hcl_loss2.item(),
                  "Validation silhouette score" : current_silhouette})

config = {
    'INIT': tune.choice(["he_normal", "he_uniform"]),
    'ENC_DR': tune.uniform(0., 0.3),
    'LR': tune.loguniform(1e-5, 2e-4),
    # 'ACT': tune.choice(["sigmoid", "tanh", "relu"]),
    'OPT': tune.choice(["Adam", "AdamW"]),
    'SOFT_NEG': tune.uniform(2., 5.),
    'HARD_NEG': tune.uniform(4., 10.),
    'TAU1': tune.choice([0.01, 0.03, 0.05, 0.07, 0.09, 0.1]),
    'TAU2': tune.choice([0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2]),
    'WD': tune.uniform(0., 0.2),
    'LAMBD': tune.uniform(0., 0.5),
    'ALPHA': tune.choice([0.1, 0.4, 0.7, 1.0])
}

scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric="Validation total loss",
    mode="min",
    max_t=2000,
    grace_period=50,
    reduction_factor=2,
)

search_alg = OptunaSearch(metric=["Training total loss", "Training hcl loss1", "Training hcl loss2", "Validation total loss", "Validation hcl loss1", "Validation hcl loss2", "Validation silhouette score"], mode=["min", "min", "min", "min", "min", "min", "max"])
os.makedirs(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/{data}_Result/{ver}/", exist_ok=True)

result = tune.run(
    fit,
    config=config,
    resources_per_trial={"cpu": 1, "gpu": 1},
    num_samples=2000,
    scheduler=scheduler,
    search_alg=search_alg,
    storage_path=f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/{data}_Result/{ver}/Hyperparameters/"
)

# best_trial = result.get_best_trial("Validation loss", "min", "last")
# print(f"Best trial config: {best_trial.config}")
# print(f"Best trial final validation loss: {best_trial.last_result['Validation loss']}")

