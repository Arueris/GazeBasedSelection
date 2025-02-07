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
        folder_name = f"{base_folder_name}_{count}"
        count += 1

    # Create the folder
    os.makedirs(folder_name)
    return folder_name

def load_model(path, type, parameter):
    model = type(**parameter)
    model.load_state_dict(torch.load(path))
    return model

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