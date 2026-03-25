import pandas as pd
import numpy as np
import random
import os
import copy

from tqdm import tqdm
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler

from datetime import datetime
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Utils import *
from Model import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = str, help = "GPU number", required=True)
parser.add_argument("--data", type = str, help = "Types of Data", required=True)
parser.add_argument("--num", type = str, help = "The project number", required=True)
parser.add_argument("--ver", type = str, help = "The version number", required=True)
parser.add_argument("--exp", type = int, help = "The experiment number", required=True)
parser.add_argument("--model_ver", type=str, help='Model Version', required=True)
parser.add_argument("--model_date", type=str, help='Model Date', required=True)
parser.add_argument("--model_num", type=int, help='Model Number', required=True)
parser.add_argument("--dim", type = int, help = "Hidden Nodes", required=True)
parser.add_argument("--opt", type = str, help = "Optimizer", default="Adam")
parser.add_argument("--init", type = str, help = "Initializer", required=True)
parser.add_argument("--act", type = str, help = "Activation", default="relu")
parser.add_argument("--dr", type = float, help = "Dropout", default=0.)
parser.add_argument("--lr", type = float, help = "Learning rate", default=1e-3)
parser.add_argument("--fac", type = float, help = "Learning rate factor", default=0.99)
# parser.add_argument("--pat", type = int, help = "Learning rate patience", default=5)
parser.add_argument("--wd", type = float, help = "Weight Decay", default=1e-1)
# parser.add_argument("--lrd", type = float, help = "Learning rate decay", default=0.5)
# parser.add_argument("--lrde", type = float, help = "Learning rate decay epoch", default=1000)
args = parser.parse_args()

import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


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

def linear_learning_rate(learning_rate, optimizer, lr_decay_rate, step):
    learning_rate = learning_rate * (lr_decay_rate ** step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    return

def get_num_nodes(data_name):
    """Returns the number of input features (genes) for a given dataset."""
    node_map = {
        "LIHC": 4781, "STAD": 4790, "LUAD": 4786,
        "LUSC": 4787, "LGG": 4789, "GSE240567": 4500
    }
    if data_name not in node_map:
        raise ValueError(f"Unknown dataset name for get_num_nodes: {data_name}")
        
    return node_map[data_name]

def fit(data, ver, model_num, date, num, experiment, train_embeddings, train_y, train_w, valid_embeddings, valid_y, valid_w, net_hparams, optim_hparams, experim_hparms, device):
    net = Classifier(net_hparams).to(device)
    ### Optimizer Setting
    ### 0-optimizer, 1-lr, 2-lr_factor, 3-lr_patience, 4-weight_decay, 5-lr_decay_rate, 6-lr_decay_epochs
    if optim_hparams[0] == "Adam":
        opt = optim.Adam(net.parameters(), lr = optim_hparams[1], betas = (0.99, 0.999), weight_decay = optim_hparams[4], amsgrad = True)
    elif optim_hparams[0] == "AdamW":
        opt = optim.AdamW(net.parameters(), lr = optim_hparams[1], betas = (0.99, 0.999), weight_decay = optim_hparams[4], amsgrad = True)
    ### Learning scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor = optim_hparams[2], patience = optim_hparams[3])

    train_loss_values, valid_loss_values = [], []
    train_auc_list, valid_auc_list = [], []
    best_valid_loss = np.inf
    best_valid_auc = 0.
    best_state = None
    opt_epoch = 1
    earlystopping_patience = experim_hparms[2]
    for epoch in tqdm(range(1, experim_hparms[0] + 1)):
        net.train()
        train_pred = net(train_embeddings)
        train_loss = F.binary_cross_entropy(train_pred, train_y, train_w)

        opt.zero_grad()
        train_loss.backward()
        opt.step()
        
        net.eval()
        with torch.no_grad():
            valid_pred = net(valid_embeddings)
            valid_loss = F.binary_cross_entropy(valid_pred, valid_y, valid_w).item()
            train_loss = train_loss.item()

        train_loss_values.append(train_loss)
        valid_loss_values.append(valid_loss)
        
        train_auc = auc(train_y, train_pred, train_w)
        valid_auc = auc(valid_y, valid_pred, valid_w)
        train_auc_list.append(train_auc)
        valid_auc_list.append(valid_auc)

        scheduler.step(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss            
            best_state = copy.deepcopy(net.state_dict())
            earlystopping_patience = experim_hparms[2]
        # if best_valid_auc < valid_auc:
        #     best_valid_auc = valid_auc
        #     best_state = copy.deepcopy(net.state_dict()) # This saves a true copy
        #     earlystopping_patience = experim_hparms[2]

        else:
        # if best_valid_loss < valid_loss:
            earlystopping_patience -= 1 ### earlystop_patience
            if (experim_hparms[1] <= epoch) & (earlystopping_patience == 0):
                print("[%s] Early stopping in [%d]" % (num, epoch))
                break

        if epoch % 200 == 0:
            plot_loss_plots(data, ver, date, num, experiment, train_loss_values, valid_loss_values)
            
    plot_loss_plots(data, ver, date, num, experiment, train_loss_values, valid_loss_values)
    plot_performance_evaluation(data, ver, date, num, experiment, train_auc_list, valid_auc_list)
    
    # Instantiate a new model (same architecture)
    opt_net = Classifier(net_hparams).to(device)
    # Load the best weights you stashed in best_state
    opt_net.load_state_dict(best_state)
    os.makedirs(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/MLP/{data}/{ver}/Saved_Model/", exist_ok=True)
    torch.save(opt_net.state_dict(), f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/MLP/{data}/{ver}/Saved_Model/[{date}_{num}]_[{experiment}]Opt_Model_StateDict.pth")
    
    return opt_net


if __name__ == "__main__":
    # parallel_grid_search()
    date = datetime.today().strftime('%m%d')
    data = args.data
    ver = args.ver
    experiment = args.exp    
    model_ver = args.model_ver
    model_date = args.model_date
    model_num = args.model_num
    num = args.num
    
    os.makedirs(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/MLP/{data}/{ver}/Saved_Model/", exist_ok=True)
    os.makedirs(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/MLP/{data}/{ver}/Learning_Curve/", exist_ok=True)
    os.makedirs(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/MLP/{data}/{ver}/Pred_Truth/", exist_ok=True)
    
    # Use CUDA when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # n_experiments = 10
    # opt_test_auc_list = []
    # final_test_auc_list = []
    # for experiment in range(1, n_experiments + 1):
    # Get Train Data, Train Label, Valid Data and Valid Label
    if data == "GSE240567":
        trainData, trainLabel, validData, validLabel, testData, testLabel = read_asthmaData(experiment, data)
        sparse_indices = load_sparse_indices(f"/home/koe3/Bioinformatics/Data/Asthma/{data}/{data}_Gene_Pathway_Mask.npz")
    else:
        trainData, trainLabel, validData, validLabel, testData, testLabel = read_cancerData(experiment, data)
        sparse_indices = load_sparse_indices(f"/home/koe3/Bioinformatics/Data/TCGA_{data}_Pathway_Mask.npz")

    train_x = torch.from_numpy(trainData.values).float().to(device, non_blocking=True)
    train_y = torch.from_numpy(trainLabel.values).float().to(device, non_blocking=True)
    valid_x = torch.from_numpy(validData.values).float().to(device, non_blocking=True)
    valid_y = torch.from_numpy(validLabel.values).float().to(device, non_blocking=True)
    test_x = torch.from_numpy(testData.values).float().to(device, non_blocking=True)
    test_y = torch.from_numpy(testLabel.values).float().to(device, non_blocking=True)

    train_sample_weight = class_weight.compute_sample_weight('balanced', trainLabel.values.ravel()).reshape(-1,1)
    valid_sample_weight = class_weight.compute_sample_weight('balanced', validLabel.values.ravel()).reshape(-1,1)
    train_w = torch.from_numpy(train_sample_weight).float().to(device, non_blocking=True)
    valid_w = torch.from_numpy(valid_sample_weight).float().to(device, non_blocking=True)
    
    # cl_net_hparams = [get_num_nodes(data), [395, 128, 64], 1, args.init, "relu", 0.10070817185386628] ### GSE240567
    # cl_net_hparams = [get_num_nodes(data), [395, 128, 32], 1, args.init, "relu", 0.1] ### LIHC
    # cl_net_hparams = [get_num_nodes(data), [395, 128, 16], 1, args.init, "sigmoid", 0.4] ### LGG
    cl_net_hparams = [get_num_nodes(data), [395, 128, 64], 1, args.init, "relu", 0.1] ### LUAD
    # Instantiate a new model (same architecture)
    pretrained_model = CLModule(cl_net_hparams, sparse_indices).to(device)
    # Load the best weights you stashed in best_state
    # pretrained_model.load_state_dict(torch.load(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/{data}_Result/{model_ver}/Saved_Model/[{model_date}_{model_num}]_[{experiment}]Opt_Model_StateDict.pth"))
    pretrained_model.load_state_dict(torch.load(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/{data}_Result/{model_ver}/Saved_Model/[{model_date}_{model_num}]_[{experiment}]Final_Model_StateDict.pth"))
    
    pretrained_model.eval()
    with torch.no_grad():
        train_cmn_embeddings, train_spf_embeddings = pretrained_model(train_x)
        valid_cmn_embeddings, valid_spf_embeddings = pretrained_model(valid_x)
        test_cmn_embeddings, test_spf_embeddings = pretrained_model(test_x)

        train_embeddings = torch.cat((train_cmn_embeddings[0], train_spf_embeddings[0]), dim=1).cpu().detach().numpy()
        valid_embeddings = torch.cat((valid_cmn_embeddings[0], valid_spf_embeddings[0]), dim=1).cpu().detach().numpy()
        test_embeddings = torch.cat((test_cmn_embeddings[0], test_spf_embeddings[0]), dim=1).cpu().detach().numpy()
        
        scaler = StandardScaler()
        scaler.fit(train_embeddings)
        train_scaled_embeddings = scaler.transform(train_embeddings)
        valid_scaled_embeddings = scaler.transform(valid_embeddings)
        test_scaled_embeddings = scaler.transform(test_embeddings)
        
        train_x_embeddings = torch.from_numpy(train_scaled_embeddings).float().to(device, non_blocking=True)
        valid_x_embeddings = torch.from_numpy(valid_scaled_embeddings).float().to(device, non_blocking=True)
        test_x_embeddings = torch.from_numpy(test_scaled_embeddings).float().to(device, non_blocking=True)

    ### MODEL PARAMETERS LOG    
    ### 0-input_nodes, 1-hidden_nodes, 2-output_nodes, 3-initializer, 4-activation, 5-dropout
    net_hparams = [train_x_embeddings.shape[1], [128, 64, args.dim], 1, args.init, args.act, args.dr]
    ### 0-optimizer, 1-lr, 2-lr_factor, 3-lr_patience, 4-weight_decay, 5-lr_decay_rate, 6-lr_decay_epochs
    optim_hparams = [args.opt, args.lr, args.fac, 10, args.wd]
    ### 0-max_epoch, 1-min_epoch, 2-earlystop_patience
    experim_hparms = [4000, 3000, 500]
    
    record = open(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/MLP/{data}/{ver}/[{date}_{experiment}]F-SHCSL_Tune_Result.txt", 'a+')
    # record.write("[%d] Trial\r\n" % (num))
    record.write("Input Nodes: %d\t\tHidden Nodes: %s\t\tOutput Nodes: %d\r\n" % (net_hparams[0], str(net_hparams[1]), net_hparams[2]))
    record.write("Initializer: %s\t\tActivation: %s\t\tDropout Rates: %s\r\n" % (net_hparams[3], net_hparams[4], net_hparams[5]))
    record.write("Optimizer: %s\t\tInit LR: %s\t\tLR Factor: %s\t\tLR Patience: %s\t\tWeight Decay: %s\r\n" % (optim_hparams[0], optim_hparams[1], optim_hparams[2], optim_hparams[3], optim_hparams[4]))
    record.write("Max Epoch: %d\t\tMin Epoch: %d\t\tEarlystopping Patience: %d\r\n" % (experim_hparms[0], experim_hparms[1], experim_hparms[2]))
    record.close()

    opt_net = fit(data, ver, model_num, date, num, experiment, train_x_embeddings, train_y, train_w, valid_x_embeddings, valid_y, valid_w, net_hparams, optim_hparams, experim_hparms, device)
    
    test_sample_weight = class_weight.compute_sample_weight('balanced', testLabel.values.ravel()).reshape(-1,1)
    test_w = torch.from_numpy(test_sample_weight).float().to(device, non_blocking=True)
    ### Evaluation
    opt_net.eval()
    with torch.no_grad():
        opt_test_pred = opt_net(test_x_embeddings)
        opt_test_auc = auc(test_y, opt_test_pred, test_w)
        # opt_test_auc_list.append(opt_test_auc)
        pd.DataFrame(np.concatenate(([opt_test_pred.cpu().detach().numpy().ravel()], [testLabel.values.ravel()]), axis=0).T).to_csv(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/MLP/{data}/{ver}/Pred_Truth/[{date}_{num}]_[{experiment}]F-SHCSL_Pred_Truth_Opt.csv", index=False)
        
    record = open(f"/home/koe3/Bioinformatics/Proposed/SHCL_v3/MLP/{data}/{ver}/[{date}_{experiment}]F-SHCSL_Tune_Result.txt", 'a+')
    # record.write("[%d] Trial\r\n" % (num))
    record.write("[%s]Best Test AUC: %.3f\r\n" % (num, opt_test_auc))
    # record.write("Final Test AUC: %.3f\r\n" % (final_test_auc))
    record.close()

    # record = open(f"[{date}_{num}]F-SHCSL_{data}_Result.txt", 'a+')
    # record.write("Average of Best Test AUC: %.3f\t\tStandard Deviation of Best Test AUC: %.4f\r\n" % (np.average(opt_test_auc_list), np.std(opt_test_auc_list)))
    # record.write("Average of Final Test AUC: %.3f\t\tStandard Deviation of Final Test AUC: %.4f\r\n" % (np.average(final_test_auc_list), np.std(final_test_auc_list)))
    # record.close()
    # print("Average of AUC: %.3f\t\tStandard Deviation of AUC: %.4f\r\n" % (np.average(opt_test_auc_list), np.std(opt_test_auc_list)))

