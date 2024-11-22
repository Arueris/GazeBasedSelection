import torch
import numpy as np
import os
from LSTM import LSTMAE
import train
import utils
import matplotlib.pyplot as plt
import seaborn as sns


def write_line_log(line, path):
    with open(path, "a") as f:
        f.write(line)

def create_plot(mse_correct, mse_incorrect, th, losses, path, name):
    fig, (ax_loss, ax_hist) = plt.subplots(2, figsize=(10,5))
    ax_loss.plot(losses)
    ax_loss.set_title("Train_loss")
    ax_loss.set_xlabel("Epoch")

    sns.histplot({"Correct": mse_correct.cpu().numpy(),
                  "Incorrect": mse_incorrect.cpu().numpy()},
                  multiple="layer", common_norm=False, stat="percent", ax=ax_hist)
    ax_hist.axvline(th, color="red", linestyle="--", label="Threshold")
    ax_hist.set_title("MSE Histogram")
    fig.suptitle(f"Model {name}")
    fig.tight_layout()
    plt.savefig(path)

def search_hyperparameter(
        lstm_hidden_dims=[32, 64, 128],
        lstm_latent_dims=[8, 16, 32, 64],
        lstm_num_layers=[1, 2, 3, 4],
        learning_rates=[1e-3]
):
    f = 83
    a = 200
    b = 500
    savefolder_all = utils.create_unique_folder("Data/HyperparameterResults")
    log_path = os.path.join(savefolder_all, "results_log.csv")
    modelfolder = utils.create_unique_folder(os.path.join(savefolder_all, "Models"))

    header = "model,cond,hidden_dim,latent_dim,num_layers,learning_rate,count_parameter,correct_acc,incorrect_acc\n"
    write_line_log(header, log_path)

    conditions = ["gaze", "headAndGaze", "nod"]

    count_models = len(lstm_hidden_dims)*len(lstm_latent_dims)*len(lstm_num_layers)*len(learning_rates)*len(conditions)
    for cond in conditions:
        savefolder = utils.create_unique_folder(os.path.join(savefolder_all, cond))
        angles_correct = np.load(f"Data/Dataset_Prepare/angles_{cond}_Correct_f{f}_b{b}_a{a}.npy")
        angles_incorrect = np.load(f"Data/Dataset_Prepare/angles_{cond}_Incorrect_f{f}_b{b}_a{a}.npy")
        names_correct = np.load(f"Data/Dataset_Prepare/names_{cond}_Correct_f{f}_b{b}_a{a}.npy")
        names_incorrect = np.load(f"Data/Dataset_Prepare/names_{cond}_Incorrect_f{f}_b{b}_a{a}.npy")
        pat_names = np.unique(names_correct)
        n = int(0.7 * len(pat_names))
        train_pats = pat_names[:n]
        test_pats = pat_names[n:]
        train_data = angles_correct[np.isin(names_correct, train_pats)]
        test_correct = angles_correct[np.isin(names_correct, test_pats)]
        test_incorrect = angles_incorrect[np.isin(names_incorrect, test_pats)]
        model_number = 0
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

if __name__=="__main__":
    search_hyperparameter()