import torch
import numpy as np
import os
from LSTM import *
from TCN import *
from VAE import *
import train
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


def write_line_log(line, path):
    with open(path, "a") as f:
        f.write(line)

def create_plot(mse_correct, mse_incorrect, th, losses, path, name):
    fig, (ax_recall, ax_hist) = plt.subplots(2, figsize=(10,5))
    # ax_loss.plot(losses)
    # ax_loss.set_title("Train_loss")
    # ax_loss.set_xlabel("Epoch")

    perc = np.linspace(80, 100, 100)
    ths = np.percentile(mse_correct.cpu().numpy(), perc)
    res = {"TH": list(), "Correct": list(), "Incorrect": list()}
    for p, t in zip(perc, ths):
        correct_acc = np.mean((mse_correct < t).cpu().numpy())
        incorrect_acc = np.mean((mse_incorrect > t).cpu().numpy())
        res["TH"].append(p)
        res["Correct"].append(correct_acc)
        res["Incorrect"].append(incorrect_acc)

    ax_recall.plot(res["TH"], res["Correct"], label="Correct", color="blue")
    ax_recall.plot(res["TH"], res["Incorrect"], label="Incorrect", color="orange")
    ax_recall.axhline(0.9, linestyle="--", color="red", alpha=.4)
    ax_recall.set_ylabel("Recall")
    ax_recall.set_xlabel("Percentile")
    ax_recall.legend()

    sns.histplot({"Correct": mse_correct.cpu().numpy(),
                  "Incorrect": mse_incorrect.cpu().numpy()},
                  multiple="layer", common_norm=False, stat="percent", ax=ax_hist)
    perc = [80, 90, 95, 99, 100]
    ths = np.percentile(mse_correct.cpu().numpy(), perc)
    for p, th in zip(perc, ths):
        ax_hist.axvline(th, color="red", linestyle="--", alpha=.4, label="Threshold_%d" % p)
    ax_hist.set_title("MSE Histogram")
    fig.suptitle(f"Model {name}")
    fig.tight_layout()
    plt.savefig(path)
    plt.close()

def search_hyperparameter(options, foldername):
    model_parameters_names = ["hidden_size", # lstm 
                        "num_channels", "kernel_size", # tcn 
                        "latent_dim"]
    log_columns = ["model", "cond"] +  model_parameters_names + ["learning_rate", "count_parameter",
                   "correct_acc", "incorrect_acc"]
    f = 62
    a = 200
    b = 500
    fps=90
    savefolder_all = utils.create_unique_folder(f"AutoEncoder/{foldername}")
    log_path = os.path.join(savefolder_all, "results_log.csv")
    modelfolder = utils.create_unique_folder(os.path.join(savefolder_all, "Models"))

    header = ",".join(log_columns)+"\n"
    write_line_log(header, log_path)

    conditions = ["gaze", "headAndGaze", "nod"]

    model_number = 0
    for cond in conditions:
        savefolder = utils.create_unique_folder(os.path.join(savefolder_all, cond))
        angles_correct = np.load(f"Data/Dataset_Prepare/angles_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}.npy")
        angles_incorrect = np.load(f"Data/Dataset_Prepare/angles_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}.npy")
        names_correct = np.load(f"Data/Dataset_Prepare/names_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}.npy")
        names_incorrect = np.load(f"Data/Dataset_Prepare/names_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}.npy")
        pat_names = np.unique(names_correct)
        n = int(0.7 * len(pat_names))
        train_pats = pat_names[:n]
        test_pats = pat_names[n:]
        train_data = angles_correct[np.isin(names_correct, train_pats)]
        test_correct = angles_correct[np.isin(names_correct, test_pats)]
        test_incorrect = angles_incorrect[np.isin(names_incorrect, test_pats)]
        for m, model_parameters, train_parameters in options:
            model = m(**model_parameters)
            model_name = f"{model.__class__.__name__}_{model_number}"
            model_number += 1
            model, losses = train.train_autoencoder(model, train_data, train_parameters["batch_size"],
                                                    train_parameters["num_epochs"], 
                                                    train_parameters["criterion"],
                                                    train_parameters["learning_rate"],
                                                    train_parameters["use_gpu"],
                                                    desc_tqdm=f"Model {model_number}/{len(options)*len(conditions)}")
            count_parameters = train.count_parameters(model)
            mse_train, mse_correct, mse_incorrect = train.test_autoencoder(train_data, test_correct, test_incorrect, model, 
                                                                           train_parameters["use_gpu"],
                                                                           train_parameters["batch_size"])
            th = np.percentile(mse_train.cpu().numpy(), 95)
            correct_acc = np.mean((mse_correct < th).cpu().numpy())
            incorrect_acc = np.mean((mse_incorrect > th).cpu().numpy())

            create_plot(mse_correct, mse_incorrect, th, losses,
                        os.path.join(savefolder, model_name + ".png"),
                        model_name)
            
            # modelline = f"{model_name},{cond},{lstm_hidden_dim},{lstm_latent_dim},{lstm_num_layer},{lr},{count_parameter},{correct_acc},{incorrect_acc}\n"
            modelline = f"{model_name},{cond}," + ",".join([f'"{model_parameters[x]}"' if x in model_parameters.keys() else 'na' for x in model_parameters_names]) + \
                        f",{train_parameters['learning_rate']},{count_parameters},{correct_acc},{incorrect_acc}\n"
            
            write_line_log(modelline, log_path)
            torch.save(model.state_dict(), os.path.join(modelfolder, f"{model_name}.pth"))


    # TODO: impelment the complete hyper parameter search for every model :-)

def search_hyperparameter_lstm(
        lstm_hidden_dims=[32, 64, 128],
        lstm_latent_dims=[8, 16, 32, 64],
        lstm_num_layers=[1, 2, 3, 4],
        learning_rates=[1e-3]
):
    f = 62
    a = 200
    b = 500
    fps=90
    savefolder_all = utils.create_unique_folder("AutoEncoder/HyperparameterResults_LSTM_fps90")
    log_path = os.path.join(savefolder_all, "results_log.csv")
    modelfolder = utils.create_unique_folder(os.path.join(savefolder_all, "Models"))

    header = "model,cond,hidden_dim,latent_dim,num_layers,learning_rate,count_parameter,correct_acc,incorrect_acc\n"
    write_line_log(header, log_path)

    conditions = ["gaze", "headAndGaze", "nod"]

    model_number = 0
    count_models = len(lstm_hidden_dims)*len(lstm_latent_dims)*len(lstm_num_layers)*len(learning_rates)*len(conditions)
    for cond in conditions:
        savefolder = utils.create_unique_folder(os.path.join(savefolder_all, cond))
        angles_correct = np.load(f"Data/Dataset_Prepare/angles_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}.npy")
        angles_incorrect = np.load(f"Data/Dataset_Prepare/angles_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}.npy")
        names_correct = np.load(f"Data/Dataset_Prepare/names_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}.npy")
        names_incorrect = np.load(f"Data/Dataset_Prepare/names_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}.npy")
        pat_names = np.unique(names_correct)
        n = int(0.7 * len(pat_names))
        train_pats = pat_names[:n]
        test_pats = pat_names[n:]
        train_data = angles_correct[np.isin(names_correct, train_pats)]
        test_correct = angles_correct[np.isin(names_correct, test_pats)]
        test_incorrect = angles_incorrect[np.isin(names_incorrect, test_pats)]
        for lstm_hidden_dim in lstm_hidden_dims:
            for lstm_latent_dim in lstm_latent_dims:
                for lstm_num_layer in lstm_num_layers:
                    for lr in learning_rates:
                        model_name = f"LSTMAE_{model_number}"
                        model_number += 1
                        model, losses = train.train_lstm_autoencoder(train_data, batch_size=1000, num_epochs=400, lstm_model=LSTMAE, num_workers=0, use_gpu=True,
                                                                     lstm_hidden_dim=lstm_hidden_dim, lstm_latent_dim=lstm_latent_dim, lstm_num_layers=lstm_num_layer,
                                                                     learning_rate=lr, desc_tqdm=f"LSTMAE {model_number}/{count_models}")
                        count_parameter = train.count_parameters(model)
                        mse_train, mse_correct, mse_incorrect = train.test_lstm_autoencoder(train_data, test_correct, test_incorrect, model)
                        th = np.percentile(mse_train.cpu().numpy(), 95)
                        correct_acc = np.mean((mse_correct < th).cpu().numpy())
                        incorrect_acc = np.mean((mse_incorrect > th).cpu().numpy())

                        create_plot(mse_correct, mse_incorrect, th, losses,
                                    os.path.join(savefolder, model_name + ".png"),
                                    model_name)
                        
                        modelline = f"{model_name},{cond},{lstm_hidden_dim},{lstm_latent_dim},{lstm_num_layer},{lr},{count_parameter},{correct_acc},{incorrect_acc}\n"
                        
                        write_line_log(modelline, log_path)
                        torch.save(model.state_dict(), os.path.join(modelfolder, f"{model_name}.pth"))


def search_hyperparameter_tcn(
        num_channels=[[8, 16], [8, 16, 32], [16, 32], [16, 32, 64]],
        kernel_sizes=[3, 5, 7],
        latent_dims=[8, 16, 32, 64],
        learning_rates=[1e-3]
    ):
    f = 62
    a = 200
    b = 500
    fps=90
    savefolder_all = utils.create_unique_folder("AutoEncoder/HyperparameterResults_TCN_fps90")
    log_path = os.path.join(savefolder_all, "results_log.csv")
    modelfolder = utils.create_unique_folder(os.path.join(savefolder_all, "Models"))

    header = "model,cond,num_channels,kernel_size,latent_dim,learning_rate,count_parameter,correct_acc,incorrect_acc\n"
    write_line_log(header, log_path)

    conditions = ["gaze", "headAndGaze", "nod"]

    model_number = 0
    count_models = len(num_channels) * len(kernel_sizes) * len(latent_dims) * len(learning_rates) * len(conditions)
    for cond in conditions:
        savefolder = utils.create_unique_folder(os.path.join(savefolder_all, cond))
        angles_correct = np.load(f"Data/Dataset_Prepare/angles_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}.npy")
        angles_incorrect = np.load(f"Data/Dataset_Prepare/angles_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}.npy")
        names_correct = np.load(f"Data/Dataset_Prepare/names_fps{fps}_{cond}_Correct_f{f}_b{b}_a{a}.npy")
        names_incorrect = np.load(f"Data/Dataset_Prepare/names_fps{fps}_{cond}_Incorrect_f{f}_b{b}_a{a}.npy")
        pat_names = np.unique(names_correct)
        n = int(0.7 * len(pat_names))
        train_pats = pat_names[:n]
        test_pats = pat_names[n:]
        train_data = angles_correct[np.isin(names_correct, train_pats)]
        test_correct = angles_correct[np.isin(names_correct, test_pats)]
        test_incorrect = angles_incorrect[np.isin(names_incorrect, test_pats)]
        for num_channel in num_channels:
            for kernel_size in kernel_sizes:
                for latent_dim in latent_dims:
                    for lr in learning_rates:
                        model_name = f"LSTMAE_{model_number}"
                        model_number += 1
                        model, losses = train.train_tcn_autoencoder(train_data, batch_size=1000, num_epochs=400,
                                                                     num_channels=num_channel, 
                                                                     kernel_size=kernel_size,
                                                                     latent_dim=latent_dim,
                                                                     learning_rate=lr, desc_tqdm=f"TCNAE {model_number}/{count_models}")
                        count_parameter = train.count_parameters(model)
                        mse_train, mse_correct, mse_incorrect = train.test_tcn_autoencoder(train_data, test_correct, test_incorrect, model)
                        th = np.percentile(mse_train.cpu().numpy(), 95)
                        correct_acc = np.mean((mse_correct < th).cpu().numpy())
                        incorrect_acc = np.mean((mse_incorrect > th).cpu().numpy())

                        create_plot(mse_correct, mse_incorrect, th, losses,
                                    os.path.join(savefolder, model_name + ".png"),
                                    model_name)
                        
                        modelline = f'{model_name},{cond},"{num_channel}",{kernel_size},{latent_dim},{lr},{count_parameter},{correct_acc},{incorrect_acc}\n'
                        write_line_log(modelline, log_path)
                        torch.save(model.state_dict(), os.path.join(modelfolder, f"{model_name}.pth"))



def create_options(
        seq_len=61,   
        channels = [[4, 8], [8, 16], [16, 32], [4, 8, 16], [8, 16, 32], [16, 32, 64]],
        kernel_sizes = [3, 5, 7],
        latent_dims = [10, 16, 24, 32],
        hidden_sizes = [10, 20, 30, 40],
        num_layers = [1],
        learning_rates = [1e-3],
        num_epochs = [400],
        batch_sizes = [1000],
        models = [TCN_VAE, LSTMAE, LSTMAE_small, TCNAE, LSTM_VAE],
        use_gpu = True):
    model_infos = {
        TCN_VAE: {"parameter": ["seq_len", "input_dim", "num_channels", "kernel_size", "latent_dim"],
                  "criterion": vae_loss},
        LSTMAE: {"parameter": ["input_dim", "hidden_size", "latent_dim", "num_layers"],
                 "criterion": nn.MSELoss()},
        LSTMAE_small: {"parameter": ["input_dim", "hidden_size", "num_layers"],
                       "criterion": nn.MSELoss()},
        TCNAE: {"parameter": ["seq_len", "input_dim", "num_channels", "kernel_size", "latent_dim"],
                "criterion": nn.MSELoss()},
        LSTM_VAE: {"parameter": ["seq_len", "input_dim", "hidden_size", "latent_dim", "num_layers"],
                   "criterion": vae_loss}
    }
    paras = {
        "seq_len": [seq_len],
        "input_dim": [1],
        "num_channels": channels,
        "kernel_size": kernel_sizes,
        "latent_dim": latent_dims,
        "hidden_size": hidden_sizes,
        "num_layers": num_layers
    }
    options = list()

    # all_paras = {
    #     "seq_len": None, "input_dim": None, "num_channels": None, "kernel_size": None, "latent_dim": None, "hidden_size": None, "num_layer": None
    # }
    for m in models:
        combinations = list(itertools.product(*(paras[key] for key in model_infos[m]["parameter"])))
        for combi in combinations:
            for lr in learning_rates:
                for num_epoch in num_epochs:
                    for batch_size in batch_sizes:
                        options.append([
                            m, 
                            {p: v for p, v in zip(model_infos[m]["parameter"], combi)},
                            {"num_epochs": num_epoch, "batch_size": batch_size, "criterion": model_infos[m]["criterion"], "learning_rate": lr, "use_gpu": use_gpu}
                        ])

    return options

def main():
    options = [
        [TCN_VAE,
         {'seq_len': 61, 'input_dim': 1, 'num_channels': [8, 16, 32], 'kernel_size': 3, 'latent_dim': 16},
         {"num_epochs": 400, "batch_size": 1000, "criterion": vae_loss, "learning_rate": 1e-3, "use_gpu": True}
        ],
        [LSTMAE, 
         {'input_dim': 1, 'hidden_size': 20, 'latent_dim': 16, 'num_layers':1}, 
         {"num_epochs": 400, "batch_size": 1000, "criterion": nn.MSELoss(), "learning_rate": 1e-3, "use_gpu": True}
        ],
        [LSTMAE_small,
         {"input_dim": 1, "hidden_size": 20, "num_layers": 1},
         {"num_epochs": 400, "batch_size": 1000, "criterion": nn.MSELoss(), "learning_rate": 1e-3, "use_gpu": True}
        ],
        [TCNAE,
         {'seq_len': 61, 'input_dim': 1, 'num_channels': [8, 16, 32], 'kernel_size': 3, 'latent_dim': 16},
         {"num_epochs": 400, "batch_size": 1000, "criterion": nn.MSELoss(), "learning_rate": 1e-3, "use_gpu": True}
        ],
        [LSTM_VAE,
         {'seq_len': 61, 'input_dim': 1, 'hidden_size': 20, 'latent_dim': 16, 'num_layers': 1},
         {"num_epochs": 400, "batch_size": 1000, "criterion": vae_loss, "learning_rate": 1e-3, "use_gpu": True}
        ],
    ]
    search_hyperparameter(options)

if __name__=="__main__":
    # main()
    # opt = create_options(
    #     seq_len=61,   
    #     channels = [[4, 8], [8, 16]],
    #     kernel_sizes = [3, 5],
    #     latent_dims = [10, 32],
    #     hidden_sizes = [10, 40],
    #     num_layers = [1],
    #     learning_rates = [1e-3],
    #     num_epochs = [30],
    #     batch_sizes = [1000],
    #     models = [TCN_VAE, LSTMAE, LSTMAE_small, TCNAE, LSTM_VAE],
    #     use_gpu = True)
    opt = create_options(
        models = [TCNAE],
        channels = [[4, 8], [8, 16], [32, 64], [16, 32, 64]],
        kernel_sizes = [3, 5, 7],
        latent_dims=[10, 14, 16, 20, 24],
        learning_rates=[1e-2, 1e-3, 1e-4, 1e-5]
    )
    search_hyperparameter(opt, "TCNAE")
    