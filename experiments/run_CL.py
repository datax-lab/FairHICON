import os
import argparse
import math
import copy

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = str, help = "GPU number", required=True)
parser.add_argument("--data", type = str, help = "Types of data", required=True)
parser.add_argument("--ver", type = str, help = "The version number", required=True)
parser.add_argument("--num", type = int, help = "The project number", required=True)
parser.add_argument("--dim", type = int, help = "Projection Dimensions", required=True)
parser.add_argument("--opt", type = str, help = "Optimizer", default="Adam")
parser.add_argument("--init", type = str, help = "Initializer", default="he_normal")
parser.add_argument("--act", type = str, help = "Activation", required=True)
parser.add_argument("--enc_dr", type = float, help = "Encoder Dropout", default=0.)
parser.add_argument("--lr", type = float, help = "Learning rate", default=1e-3)
parser.add_argument("--fac", type = float, help = "Learning rate factor", default=0.99)
# parser.add_argument("--pat", type = int, help = "Learning rate patience", default=5)
parser.add_argument("--wd", type = float, help = "Weight Decay", default=1e-1)
parser.add_argument("--tau1", type = float, help = "Tau1", default=0.07)
parser.add_argument("--tau2", type = float, help = "Tau2", default=0.07)
parser.add_argument("--lamb", type = float, help = "Lambda", default=0.5)
parser.add_argument("--sn", type = float, help = "Soft Negative Weight", default=3.)
parser.add_argument("--hn", type = float, help = "Hard Negative Weight", default=5.)

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from Adapt_Proto_Utils_Hierarchy import *
from Model import *
from Adapt_Proto_CL_Hierarchy_v2 import *

import warnings
warnings.filterwarnings('ignore')

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

def get_num_nodes(data):
    ### number of genes
    if data == "LIHC":
        In_Nodes = 4781
    elif data == "STAD":
        In_Nodes = 4790
    elif data == "LUAD":
        In_Nodes = 4786
    elif data == "LUSC":
        In_Nodes = 4787
    elif data == "LGG":
        In_Nodes = 4789
    elif data == "GSE240567":
        In_Nodes = 4500
    
    return In_Nodes

def linear_learning_rate(learning_rate, optimizer, lr_decay_rate, step):

    learning_rate = learning_rate * (lr_decay_rate ** step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    return

def get_common_groups(labels, sexs):
    common_groups = []
    
    for lbl, sex in zip(labels, sexs):
        
        if lbl==1 and sex==1:
            common_groups.append("Common M LTS")
        
        elif lbl==1 and sex==0:
            common_groups.append("Common F LTS")
        
        elif lbl==0 and sex==1:
            common_groups.append("Common M non_LTS")
        
        elif lbl==0 and sex==0:
            common_groups.append("Common F non_LTS")
            
    return common_groups


def get_sex_specific_groups(labels, sexs):
    specific_groups = []
    
    for lbl, sex in zip(labels, sexs):
        
        if lbl==1 and sex==1:
            specific_groups.append("Male LTS")
        
        elif lbl==1 and sex==0:
            specific_groups.append("Female LTS")
        
        elif lbl==0 and sex==1:
            specific_groups.append("Male non_LTS")
        
        elif lbl==0 and sex==0:
            specific_groups.append("Female non_LTS")
            
    return specific_groups


def fit(data, ver, date, num, experiment, train_x, train_y, valid_x, valid_y, test_x, test_y, sparse_indices, net_hparams, optim_hparams, experim_hparms, device):
    """
        Terminology:
            lts: long term survival
            sts: short term survival
            cmn: common
            ml: male
            fml: female
            es: early stopping
    """    
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=optim_hparams[2], patience=optim_hparams[3])
    
    train_total_loss_values, train_hcl_loss1_values, train_hcl_loss2_values = [], [], []
    valid_total_loss_values, valid_hcl_loss1_values, valid_hcl_loss2_values = [], [], []
    lambd = optim_hparams[7]
    best_valid_loss = np.inf
    best_state = net.state_dict()
    opt_epoch = 1
    earlystopping_patience = experim_hparms[2]

    for epoch in range(1, experim_hparms[0] + 1):
        # torch.cuda.empty_cache()
        # if (optim_hparams[6] != -1) and (epoch % optim_hparams[6] == 0):
        #     linear_learning_rate(optim_hparams[1], opt, optim_hparams[5], (epoch / optim_hparams[6]))                    
        net.train()
        train_cmn_embeddings, train_spf_embeddings = net(train_x)
        train_hcl_loss1, train_hcl_loss2 = FairContrastiveLearning(train_cmn_embeddings[1], train_spf_embeddings[1],
                                                                   train_mask_lts, train_mask_nlts, train_mask_ml, train_mask_fml,
                                                                   optim_hparams[5], optim_hparams[6], optim_hparams[8], optim_hparams[9])

        train_total_loss = train_hcl_loss1 + (lambd * train_hcl_loss2)
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
        
        net.eval()
        with torch.no_grad():
            valid_cmn_embeddings, valid_spf_embeddings = net(valid_x)
            valid_hcl_loss1, valid_hcl_loss2 = FairContrastiveLearning(valid_cmn_embeddings[1], valid_spf_embeddings[1],
                                                                       valid_mask_lts, valid_mask_nlts, valid_mask_ml, valid_mask_fml,
                                                                       optim_hparams[5], optim_hparams[6], optim_hparams[8], optim_hparams[9])
            
            valid_total_loss = (valid_hcl_loss1 + (lambd * valid_hcl_loss2)).item()
            train_total_loss = train_total_loss.item()
            
            train_hcl_loss1_values.append(train_hcl_loss1.item())
            valid_hcl_loss1_values.append(valid_hcl_loss1.item())

            train_hcl_loss2_values.append((lambd * train_hcl_loss2).item())
            valid_hcl_loss2_values.append((lambd * valid_hcl_loss2).item())

            train_total_loss_values.append(train_total_loss)
            valid_total_loss_values.append(valid_total_loss)
            
            scheduler.step(valid_total_loss)
            if valid_total_loss < best_valid_loss:
                best_valid_loss = valid_total_loss
                # best_state = net.state_dict()
                best_state = copy.deepcopy(net.state_dict()) # This saves a true copy
                opt_epoch = epoch
                earlystopping_patience = experim_hparms[2]

            else:
                earlystopping_patience -= 1 ### earlystop_patience
                if (experim_hparms[1] <= epoch) & (earlystopping_patience == 0):
                    print("[%s] Early stopping in [%d]" % (num, epoch))
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
        torch.save(opt_net, f"/home/koe3/Bioinformatics/Proposed/SHCL_v1/{data}_Result/{ver}/Saved_Model/[{date}_{num}]_[{experiment}]Opt_Model.pt")
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
        torch.save(net, f"/home/koe3/Bioinformatics/Proposed/SHCL_v1/{data}_Result/{ver}/Saved_Model/[{date}_{num}]_[{experiment}]Final_Opt_Model.pt")
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
    
    return


if __name__ == "__main__":
    date = datetime.today().strftime('%m%d')
    data = args.data
    ver = args.ver
    num = args.num
    # Use CUDA when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Populate In_Nodes based on the type of data
    In_Nodes = get_num_nodes(data)
    if data == "GSE240567":
        sparse_indices = load_sparse_indices(f"/home/koe3/Bioinformatics/Data/Asthma/{data}/{data}_Gene_Pathway_Mask.npz")
    else:
        sparse_indices = load_sparse_indices(f"/home/koe3/Bioinformatics/Data/TCGA_{data}_Pathway_Mask.npz")
    ### MODEL PARAMETERS LOG
    ### 0-input_nodes, 1-hidden_nodes, 2-output_nodes, 3-initializer, 4-activation, 5-encoder_dropout
    net_hparams = [In_Nodes, [395, 128, args.dim], 1, args.init, args.act, args.enc_dr]
    ### 0-optimizer, 1-lr, 2-lr_factor, 3-lr_patience, 4-weight_decay, 5-tau1, 6-tau2, 7-lambda, 8-sg, 9-hn
    optim_hparams = [args.opt, args.lr, args.fac, 10, args.wd, args.tau1, args.tau2, args.lamb, args.sn, args.hn]
    ### 0-max_epoch, 1-min_epoch, 2-earlystop_patience
    experim_hparms = [3000, 2000, 500]
    
    os.makedirs(f"{data}_Result/{ver}/", exist_ok=True)
    os.makedirs(f"{data}_Result/{ver}/Saved_Model/", exist_ok=True)
    os.makedirs(f"{data}_Result/{ver}/Learning_Curve/", exist_ok=True)
    os.makedirs(f"{data}_Result/{ver}/Evaluation_Plot/", exist_ok=True)
    os.makedirs(f"{data}_Result/{ver}/Pred_Truth/", exist_ok=True)
    os.makedirs(f"{data}_Result/{ver}/Embeddings_Plot/", exist_ok=True)
    
    record = open(f"{data}_Result/{ver}/[{date}_{num}]F_SHCSL_{data}_Result.txt", 'a+')
    record.write("Input Nodes: %d\t\tHidden Nodes: %s\t\tOutput Nodes: %d\r\n" % (net_hparams[0], str(net_hparams[1]), net_hparams[2]))
    record.write("Initializer: %s\t\tActivation: %s\t\tEncoder Dropout Rates: %s\r\n" % (net_hparams[3], net_hparams[4], net_hparams[5]))
    record.write("Optimizer: %s\t\tInit LR: %s\t\tLR Factor: %s\t\tLR Patience: %s\t\tWeight Decay: %s\r\n" % (optim_hparams[0], optim_hparams[1], optim_hparams[2], optim_hparams[3], optim_hparams[4]))
    record.write("TAU1: %s\t\tTAU2: %s\t\tLAMBDA: %s\t\tSN: %s\t\tHN: %s\r\n" % (optim_hparams[5], optim_hparams[6], optim_hparams[7], optim_hparams[8], optim_hparams[9]))
    record.write("Max Epoch: %d\t\tMin Epoch: %d\t\tEarlystopping Patience: %d\r\n" % (experim_hparms[0], experim_hparms[1], experim_hparms[2]))
    record.close()
    
    n_experiments = 10
    # test_auc_list = []
    # final_test_auc_list = []
    for experiment in range(1, n_experiments + 1):
        if data == "GSE240567":
            trainData, trainLabel, validData, validLabel, testData, testLabel = read_asthmaData(experiment, data)
        else:
            trainData, trainLabel, validData, validLabel, testData, testLabel = read_cancerData(experiment, data)
        # Get Train Data, Train Label, Valid Data and Valid Label
        train_x = torch.from_numpy(trainData.values).float().to(device, non_blocking=True)
        train_y = torch.from_numpy(trainLabel.values).float().to(device, non_blocking=True)
        valid_x = torch.from_numpy(validData.values).float().to(device, non_blocking=True)
        valid_y = torch.from_numpy(validLabel.values).float().to(device, non_blocking=True)
        test_x = torch.from_numpy(testData.values).float().to(device, non_blocking=True)
        test_y = torch.from_numpy(testLabel.values).float().to(device, non_blocking=True)
        
        fit(data, ver, date, num, experiment, train_x, train_y, valid_x, valid_y, test_x, test_y, sparse_indices, net_hparams, optim_hparams, experim_hparms, device)
        
        