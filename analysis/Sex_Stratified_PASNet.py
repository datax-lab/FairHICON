import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import random
import os
import copy      # ADDED: required for copy.deepcopy
import pickle    # ADDED: required for pickle.dump

from scipy import sparse
from tqdm import tqdm
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from Model import *
from datetime import datetime
date = datetime.today().strftime('%m%d')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = str, help = "GPU number", required=True)
parser.add_argument("--data", type = str, help = "Types of Cancer", required=True)
parser.add_argument("--num", type = int, help = "The project number", required=True)
parser.add_argument("--epoch", type = int, help = "Number of Epochs", default=300)
args = parser.parse_args() # ADDED: parser must be parsed!

import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Use CUDA when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def get_num_nodes(data_name):
    """Returns the number of input features (genes) for a given dataset."""
    node_map = {
        "LIHC": 4781, "LUAD": 4786,
        "LGG": 4789, "GSE240567": 4500
    }
    if data_name not in node_map:
        raise ValueError(f"Unknown dataset name for get_num_nodes: {data_name}")
        
    return node_map[data_name]

# FIXED: Removed the .cpu() checks since we are explicitly passing numpy arrays in Step 3
def auc(y_true, y_pred, sample_weight = None):
    if sample_weight is None:
        auc_score = roc_auc_score(y_true.ravel(), y_pred.ravel())
    else:
        auc_score = roc_auc_score(y_true.ravel(), y_pred.ravel(), sample_weight = sample_weight)
    return auc_score

def integrate(x, y):
    area = 0
    for i in range(1, len(x)):
        h = x[i-1] - x[i]
        area += h * (y[i] + y[i-1]) / 2
    return area

# FIXED: Removed the .cpu() checks since inputs are numpy arrays
def auprc(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
        auprc_score = integrate(recall, precision)
    else:
        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel(), sample_weight=sample_weight)
        auprc_score = integrate(recall, precision)
    return auprc_score

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

num = args.num
data = args.data
epochs = args.epoch

female_test_auc_list = []
male_test_auc_list = []
female_test_auprc_list = []
male_test_auprc_list = []

# ADDED: Lists to store overall combined metrics
overall_test_auc_list = []
overall_test_auprc_list = []

for experiment in range(1, 11):    
    if data == "GSE240567":
        sparse_indices = load_sparse_indices(f"/home/koe3/Bioinformatics/Data/Asthma/{data}/{data}_Gene_Pathway_Mask.npz")
        trainData, trainLabel, validData, validLabel, testData, testLabel = read_asthmaData(experiment, data)
    else:
        sparse_indices = load_sparse_indices(f"/home/koe3/Bioinformatics/Data/TCGA_{data}_Pathway_Mask.npz")
        trainData, trainLabel, validData, validLabel, testData, testLabel = read_cancerData(experiment, data)
    
    # FIXED: using correct indices from the pandas dataframes, not the undefined tensors
    female_trainData = trainData[trainData["Sex"] == 0].iloc[:,:-1]
    male_trainData = trainData[trainData["Sex"] == 1].iloc[:,:-1]
    female_trainLabel = trainLabel.loc[female_trainData.index]
    male_trainLabel = trainLabel.loc[male_trainData.index]

    female_validData = validData[validData["Sex"] == 0].iloc[:,:-1]
    male_validData = validData[validData["Sex"] == 1].iloc[:,:-1]
    female_validLabel = validLabel.loc[female_validData.index]
    male_validLabel = validLabel.loc[male_validData.index]

    female_testData = testData[testData["Sex"] == 0].iloc[:,:-1]
    male_testData = testData[testData["Sex"] == 1].iloc[:,:-1]
    female_testLabel = testLabel.loc[female_testData.index]
    male_testLabel = testLabel.loc[male_testData.index]

    # FIXED: changed 'cancer' to 'data'
    net_hparams = [get_num_nodes(data), [395, 128, 64], 1, "he_uniform"]

    female_train_x = torch.from_numpy(female_trainData.values).float().to(device, non_blocking=True)
    male_train_x = torch.from_numpy(male_trainData.values).float().to(device, non_blocking=True)
    female_train_y = torch.from_numpy(female_trainLabel.values).float().to(device, non_blocking=True)
    male_train_y = torch.from_numpy(male_trainLabel.values).float().to(device, non_blocking=True)

    female_valid_x = torch.from_numpy(female_validData.values).float().to(device, non_blocking=True)
    male_valid_x = torch.from_numpy(male_validData.values).float().to(device, non_blocking=True)
    female_valid_y = torch.from_numpy(female_validLabel.values).float().to(device, non_blocking=True)
    male_valid_y = torch.from_numpy(male_validLabel.values).float().to(device, non_blocking=True)

    female_test_x = torch.from_numpy(female_testData.values).float().to(device, non_blocking=True)
    male_test_x = torch.from_numpy(male_testData.values).float().to(device, non_blocking=True)
    female_test_y = torch.from_numpy(female_testLabel.values).float().to(device, non_blocking=True)
    male_test_y = torch.from_numpy(male_testLabel.values).float().to(device, non_blocking=True)

    print(f"--- Running Naive Sex-Stratified PASNet Baseline (Exp {experiment}) ---")
    
    model_male = PASNet(net_hparams, sparse_indices).to(device)
    model_female = PASNet(net_hparams, sparse_indices).to(device)

    optimizer_m = optim.Adam(model_male.parameters(), lr=0.001)
    optimizer_f = optim.Adam(model_female.parameters(), lr=0.001)
    scheduler_m = optim.lr_scheduler.ReduceLROnPlateau(optimizer_m, 'min', factor = 0.95, patience = 100)
    scheduler_f = optim.lr_scheduler.ReduceLROnPlateau(optimizer_f, 'min', factor = 0.95, patience = 100)

    def train_independent_model(model, optimizer, scheduler, X_train, y_train, X_valid, y_valid, epochs=50):
        train_loss_values = []
        valid_loss_values = []
        best_valid_loss = float('inf')
        opt_net = None
        for epoch in tqdm(range(1, epochs + 1)):
            model.train()
            pred = model(X_train)
            loss = F.binary_cross_entropy(pred, y_train)
            
            # FIXED: changed 'opt' to 'optimizer'
            optimizer.zero_grad()
            loss.backward()
            
            if model.layer1.weight.grad is not None:
                model.layer1.weight.grad = fixed_s_mask(model.layer1.weight.grad, sparse_indices)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_valid)
                val_loss = F.binary_cross_entropy(val_pred, y_valid).item()

                train_loss_values.append(loss.item())
                valid_loss_values.append(val_loss)

            # FIXED: changed 'valid_loss' to 'val_loss'
            scheduler.step(val_loss)
            
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                opt_net = copy.deepcopy(model)

            # FIXED: changed 'cancer' to 'data'
            if epoch % 10 == 0:
                with open(f'Intermediate_Loss_List/[{date}_{num}]_[{experiment}]{data}_train_loss.pkl', 'wb') as f_train :
                    pickle.dump(train_loss_values, f_train)
                with open(f'Intermediate_Loss_List/[{date}_{num}]_[{experiment}]{data}_valid_loss.pkl', 'wb') as f_valid :
                    pickle.dump(valid_loss_values, f_valid)

        return opt_net

    print("Training Male-only PASNet...")
    # FIXED: Added male_valid_x, male_valid_y arguments
    model_male = train_independent_model(model_male, optimizer_m, scheduler_m, male_train_x, male_train_y, male_valid_x, male_valid_y, epochs)

    print("Training Female-only PASNet...")
    # FIXED: Added female_valid_x, female_valid_y arguments
    model_female = train_independent_model(model_female, optimizer_f, scheduler_f, female_train_x, female_train_y, female_valid_x, female_valid_y, epochs)

    model_male.eval()
    model_female.eval()

    with torch.no_grad():
        preds_male = model_male(male_test_x).squeeze().cpu().numpy()
        preds_female = model_female(female_test_x).squeeze().cpu().numpy()

        y_test_m_np = male_test_y.cpu().numpy()
        y_test_f_np = female_test_y.cpu().numpy()

    # Calculate cohort-specific metrics
    auroc_male = auc(y_test_m_np, preds_male)
    auroc_female = auc(y_test_f_np, preds_female)

    print(f"Male-Only PASNet   -> Male Test AUROC: {auroc_male:.3f}")
    print(f"Female-Only PASNet -> Female Test AUROC: {auroc_female:.3f}")
    male_test_auc_list.append(auroc_male)
    female_test_auc_list.append(auroc_female)

    auprc_male = auprc(y_test_m_np, preds_male)
    auprc_female = auprc(y_test_f_np, preds_female)

    print(f"Male-Only PASNet   -> Male Test AUPRC: {auprc_male:.3f}")
    print(f"Female-Only PASNet -> Female Test AUPRC: {auprc_female:.3f}")
    male_test_auprc_list.append(auprc_male)
    female_test_auprc_list.append(auprc_female)
    
    # ---------------------------------------------------------
    # ADDED: Calculate "Overall" Combined Performance for Table
    # ---------------------------------------------------------
    combined_y_test = np.concatenate([y_test_m_np, y_test_f_np])
    combined_preds = np.concatenate([preds_male, preds_female])

    overall_auroc = auc(combined_y_test, combined_preds)
    overall_auprc = auprc(combined_y_test, combined_preds)
    
    print(f"Stratified Baseline -> Overall Test AUROC: {overall_auroc:.3f}")
    print(f"Stratified Baseline -> Overall Test AUPRC: {overall_auprc:.3f}\n")
    
    overall_test_auc_list.append(overall_auroc)
    overall_test_auprc_list.append(overall_auprc)

# FIXED: Added Overall output formatting to file
record = open(f"Results/[{date}_{num}]PASNet_{data}_Result.txt", 'a+')
record.write("Average of Male Test AUC: %.3f\t\tStandard Deviation of Male Test AUC: %.4f\r\n" % (np.average(male_test_auc_list), np.std(male_test_auc_list)))
record.write("Average of Female Test AUC: %.3f\t\tStandard Deviation of Female Test AUC: %.4f\r\n" % (np.average(female_test_auc_list), np.std(female_test_auc_list)))
record.write("Average of Male Test AUPRC: %.3f\t\tStandard Deviation of Male Test AUPRC: %.4f\r\n" % (np.average(male_test_auprc_list), np.std(male_test_auprc_list)))
record.write("Average of Female Test AUPRC: %.3f\t\tStandard Deviation of Female Test AUPRC: %.4f\r\n" % (np.average(female_test_auprc_list), np.std(female_test_auprc_list)))

record.write("\r\n--- REQUIRED FOR REBUTTAL (OVERALL PERFORMANCE) ---\r\n")
record.write("Average of Overall Test AUC: %.3f\t\tStandard Deviation of Overall Test AUC: %.4f\r\n" % (np.average(overall_test_auc_list), np.std(overall_test_auc_list)))
record.write("Average of Overall Test AUPRC: %.3f\t\tStandard Deviation of Overall Test AUPRC: %.4f\r\n" % (np.average(overall_test_auprc_list), np.std(overall_test_auprc_list)))
record.close()