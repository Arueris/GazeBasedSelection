import os
import torch
import numpy as np
from LSTM import LSTMAE, LSTMAE_small
from TCN import TCNAE
from VAE import LSTM_VAE, TCN_VAE

def create_unique_folder(base_folder_name):
    # Initialize the folder name
    folder_name = base_folder_name
    count = 1

    # While a folder with this name exists, modify the name
    while os.path.exists(folder_name):
        if len(os.listdir(folder_name)) == 0:
            return folder_name
        folder_name = f"{base_folder_name}_{count}"
        count += 1

    # Create the folder
    os.makedirs(folder_name)
    return folder_name

def load_model(path, type, parameter):
    model = type(**parameter)
    model.load_state_dict(torch.load(path))
    return model

def load_raw_data(fps, cond, f, b, a, d_type="vel", pre_path="../"):
    if d_type=="angles":
        angles_correct = np.load(f"{pre_path}Data/Dataset_Prepare/angles_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}_finalRounds.npy")
        angles_incorrect = np.load(f"{pre_path}Data/Dataset_Prepare/angles_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}_finalRounds.npy")
        names_correct = np.load(f"{pre_path}Data/Dataset_Prepare/names_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}_finalRounds.npy")
        names_incorrect = np.load(f"{pre_path}Data/Dataset_Prepare/names_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}_finalRounds.npy")
    elif d_type=="vel":
        angles_correct = np.load(f"{pre_path}Data/Dataset_Prepare/anglesVel_fps{fps}_{cond}_Correct_b{b}_a{a}_finalRounds.npy")
        angles_incorrect = np.load(f"{pre_path}Data/Dataset_Prepare/anglesVel_fps{fps}_{cond}_Incorrect_b{b}_a{a}_finalRounds.npy")
        names_correct = np.load(f"{pre_path}Data/Dataset_Prepare/namesVel_fps{fps}_{cond}_Correct_b{b}_a{a}_finalRounds.npy")
        names_incorrect = np.load(f"{pre_path}Data/Dataset_Prepare/namesVel_fps{fps}_{cond}_Incorrect_b{b}_a{a}_finalRounds.npy")
    elif d_type=="unityVel":
        angles_correct = np.load(f"{pre_path}Data/Dataset_Prepare/unity_anglesVel_fps{fps}_{cond}_Correct_b{b}_a{a}_finalRounds.npy")
        angles_incorrect = np.load(f"{pre_path}Data/Dataset_Prepare/unity_anglesVel_fps{fps}_{cond}_Incorrect_b{b}_a{a}_finalRounds.npy")
        names_correct = np.load(f"{pre_path}Data/Dataset_Prepare/unity_namesVel_fps{fps}_{cond}_Correct_b{b}_a{a}_finalRounds.npy")
        names_incorrect = np.load(f"{pre_path}Data/Dataset_Prepare/unity_namesVel_fps{fps}_{cond}_Incorrect_b{b}_a{a}_finalRounds.npy")
    else:
        raise ValueError(f"Invalid d_type {d_type}")
    return angles_correct, angles_incorrect, names_correct, names_incorrect

def load_data(fps, cond, f, b, a, train_portion=0.7, d_type="vel", pre_path="../"):
    """
    Load the dataset for training and testing.
    Parameters:
        fps (int): Frames per second.
        cond (str): Condition name.
        f (int): Feature dimension.
        b (int): time before selection
        a (int): time after selection
        train_portion (float): Portion of data to use for training.
        d_type (str): Type of data ("angles" or "vel").
        pre_path (str): Pre-path for loading data files.
    """
    if d_type=="angles":
        angles_correct = np.load(f"{pre_path}Data/Dataset_Prepare/angles_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}_finalRounds.npy")
        angles_incorrect = np.load(f"{pre_path}Data/Dataset_Prepare/angles_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}_finalRounds.npy")
        names_correct = np.load(f"{pre_path}Data/Dataset_Prepare/names_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}_finalRounds.npy")
        names_incorrect = np.load(f"{pre_path}Data/Dataset_Prepare/names_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}_finalRounds.npy")
    elif d_type=="vel":
        angles_correct = np.load(f"{pre_path}Data/Dataset_Prepare/anglesVel_fps{fps}_{cond}_Correct_b{b}_a{a}_finalRounds.npy")
        angles_incorrect = np.load(f"{pre_path}Data/Dataset_Prepare/anglesVel_fps{fps}_{cond}_Incorrect_b{b}_a{a}_finalRounds.npy")
        names_correct = np.load(f"{pre_path}Data/Dataset_Prepare/namesVel_fps{fps}_{cond}_Correct_b{b}_a{a}_finalRounds.npy")
        names_incorrect = np.load(f"{pre_path}Data/Dataset_Prepare/namesVel_fps{fps}_{cond}_Incorrect_b{b}_a{a}_finalRounds.npy")
    else:
        raise ValueError(f"Invalid d_type {d_type}")
    pat_names = np.unique(names_correct)
    n = int(train_portion * len(pat_names))
    train_pats = pat_names[:n]
    test_pats = pat_names[n:]
    train_data = angles_correct[np.isin(names_correct, train_pats)]
    test_correct = angles_correct[np.isin(names_correct, test_pats)]
    test_incorrect = angles_incorrect[np.isin(names_incorrect, test_pats)]
    return train_data, test_correct, test_incorrect #, {'Names_correct': names_correct, 'Names_incorrect': names_incorrect, 'angles_correct': angles_correct, 'angles_incorrect': angles_incorrect, 'Train_pats': train_pats, 'Test_pats': test_pats}

def load_np_dataset(fps, cond, f, b, a):
    angles_correct = np.load(f"Data/Dataset_Prepare/angles_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}.npy")
    angles_incorrect = np.load(f"Data/Dataset_Prepare/angles_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}.npy")
    names_correct = np.load(f"Data/Dataset_Prepare/names_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}.npy")
    names_incorrect = np.load(f"Data/Dataset_Prepare/names_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}.npy")
    return angles_correct, angles_incorrect, names_correct, names_incorrect   

def try_float(x):
    try:
        x = float(x)
    except:
        pass
    return x

def parese_model_info(path):
    res = dict()
    with open(path, "r") as f:
        lines = f.readlines()
    for l in lines:
        l = l.replace("\n", "")
        l_split = l.split("\t")
        if "," not in l_split[1]:
            res[l_split[0]] = try_float(l_split[1])
        else:
            para_split = l_split[1].split(",")
    return lines

if __name__=="__main__":
    model_infos = parese_model_info("AutoEncoder/Results/TCNAE_CV/Models/TCNAE_480_headAndGaze_info.txt")
    print(model_infos)