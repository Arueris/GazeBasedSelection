from LSTM import LSTMAutoencoder, LSTMAE, LSTMAE_small
from VAE import LSTM_VAE, TCN_VAE, vae_loss
from TCN import TCNAE
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GazeBasedInteraction(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_autoencoder(model, train_data, batch_size, num_epochs, criterion, learning_rate=1e-3, use_gpu=False, num_worker=0, desc_tqdm=None):
    dataset = GazeBasedInteraction(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)
    device = torch.device("cuda") if torch.cuda.is_available() and use_gpu else torch.device("cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = list()
    model.train()
    desc = desc_tqdm if desc_tqdm is not None else "Train Autoencoder"
    pbar = tqdm(range(num_epochs), desc=desc)
    for _ in pbar:
        runningLoss = 0
        n = 0
        for x in dataloader:
            optimizer.zero_grad()
            x = x.type(torch.float)
            if isinstance(model, (LSTMAE, LSTM_VAE, LSTMAE_small)):
                x = x[:, :, None]
            if isinstance(model, (TCNAE, TCN_VAE)):
                x = x[:, None, :]
            x = x.to(device)

            if isinstance(model, (LSTM_VAE, TCN_VAE)):
                output, mu, log_var = model(x)
                loss = criterion(output, x, mu, log_var)
                # print(f"{mu.shape=}")
                # print(f"{log_var.shape=}")
            else:
                output = model(x)
                loss = criterion(output, x)

            # print(loss)
            # Compute reconstruction loss
            # loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            n += 1
        pbar.set_postfix(Loss=runningLoss/n)
        losses.append(runningLoss/n)
    return model, losses


def test_autoencoder(train_samples, correct_samples, incorrect_samples, model, use_gpu=True, batch_size=1000, num_workers=0):
    dataset_correct = GazeBasedInteraction(correct_samples)
    # dataloader_correct = DataLoader(dataset_correct, batch_size=batch_size, shuffle=False, num_wokers=num_workers)
    dataset_incorrect = GazeBasedInteraction(incorrect_samples)
    # dataloader_incorrect = DataLoader(dataset_incorrect, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataset_train = GazeBasedInteraction(train_samples)
    dataloader = {
        "Train": DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "Correct": DataLoader(dataset_correct, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "Incorrect": DataLoader(dataset_incorrect, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    }
    
    device = torch.device("cuda") if torch.cuda.is_available() and use_gpu else torch.device("cpu")
    model = model.to(device)
    model.eval()
    mses = dict()
    with torch.no_grad():
        for k in ["Train", "Correct", "Incorrect"]:
            mses[k] = list()
            for x in dataloader[k]:
                x = x.type(torch.float)
                if isinstance(model, (LSTMAE, LSTM_VAE, LSTMAE_small)):
                    x = x[:, :, None]
                if isinstance(model, (TCNAE, TCN_VAE)):
                    x = x[:, None, :]
                x = x.to(device)
                if isinstance(model, (LSTM_VAE, TCN_VAE)):
                    out, _, _ = model(x)
                else:
                    out = model(x)
                # out = model(x)
                mse = torch.mean((out - x)**2, dim=(1, 2))
                mses[k].append(mse)
            mses[k] = torch.concat(mses[k]) if len(mses[k]) > 0 else None
    return mses["Train"], mses["Correct"], mses["Incorrect"]



def train_lstm_autoencoder(train_data, batch_size, num_epochs=200, num_workers=0, lstm_model=LSTMAutoencoder, lstm_hidden_dim=64, lstm_latent_dim=16, lstm_num_layers=1, learning_rate=1e-3, use_gpu=False, desc_tqdm=None):
    input_dim = 1       # Number of features (for univariate time series)
    # hidden_dim = 64     # Number of hidden units in LSTM layers
    # latent_dim = 16     # Size of the latent vector
    # num_layers = 1     # Number of LSTM layers
    # learning_rate = 1e-3
    # num_epochs = 200

    # train_data = torch.tensor(train_data).type(torch.float)[:,:,None]

    dataset = GazeBasedInteraction(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda") if torch.cuda.is_available() and use_gpu else torch.device("cpu")
    # device = torch.device("cpu")
    # Instantiate model, define loss and optimizer
    model = lstm_model(input_dim, lstm_hidden_dim, lstm_latent_dim, lstm_num_layers)
    model = model.to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy training data (replace with your own time series)
    # Shape: (batch_size, sequence_length, input_dim)
    # train_data = torch.randn(100, 30, input_dim)  # 100 sequences of 30 time steps

    # train_data = train_data.to(device)
    losses = list()
    model.train()
    # Training loop
    desc = desc_tqdm if desc_tqdm is not None else "LSTM Autoencoder"
    pbar = tqdm(range(num_epochs), desc=desc)
    for _ in pbar:
        # pbar = tqdm(dataloader, desc=f"LSTM AutoEncoder Epoch {epoch+1}/{num_epochs}")
        runningLoss = 0
        n = 0
        for x in dataloader:
            optimizer.zero_grad()
            x = x.type(torch.float)[:, :, None].to(device)
            # print(x.shape)
            # Forward pass
            output = model(x)
            
            # Compute reconstruction loss
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            n += 1
        pbar.set_postfix(Loss=loss.item())
        losses.append(runningLoss/n)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return model, losses


def test_lstm_autoencoder(train_samples, correct_samples, incorrect_samples, model, use_gpu=True, batch_size=1000, num_workers=0):
    dataset_correct = GazeBasedInteraction(correct_samples)
    # dataloader_correct = DataLoader(dataset_correct, batch_size=batch_size, shuffle=False, num_wokers=num_workers)
    dataset_incorrect = GazeBasedInteraction(incorrect_samples)
    # dataloader_incorrect = DataLoader(dataset_incorrect, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataset_train = GazeBasedInteraction(train_samples)
    dataloader = {
        "Train": DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "Correct": DataLoader(dataset_correct, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "Incorrect": DataLoader(dataset_incorrect, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    device = torch.device("cuda") if torch.cuda.is_available() and use_gpu else torch.device("cpu")
    model = model.to(device)
    model.eval()
    mses = dict()
    with torch.no_grad():
        for k in ["Train", "Correct", "Incorrect"]:
            mses[k] = list()
            for x in dataloader[k]:
                x = x.type(torch.float)[:, :, None].to(device)
                out = model(x)
                mse = torch.mean((out - x)**2, dim=(1, 2))
                mses[k].append(mse)
            mses[k] = torch.concat(mses[k])
    return mses["Train"], mses["Correct"], mses["Incorrect"]


def train_tcn_autoencoder(train_data, batch_size=1000, num_epochs=200, num_workers=0, tcn_model=TCNAE, 
                          num_channels=[16, 32, 64], kernel_size=3, latent_dim=16, learning_rate=1e-3, 
                          use_gpu=False, desc_tqdm=None):
    input_dim = 1       # Number of features (for univariate time series)
    # hidden_dim = 64     # Number of hidden units in LSTM layers
    # latent_dim = 16     # Size of the latent vector
    # num_layers = 1     # Number of LSTM layers
    # learning_rate = 1e-3
    # num_epochs = 200

    # train_data = torch.tensor(train_data).type(torch.float)[:,:,None]

    dataset = GazeBasedInteraction(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda") if torch.cuda.is_available() and use_gpu else torch.device("cpu")
    # device = torch.device("cpu")
    # Instantiate model, define loss and optimizer
    model = tcn_model(train_data.shape[1], input_dim, num_channels, kernel_size, latent_dim)
    model = model.to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy training data (replace with your own time series)
    # Shape: (batch_size, sequence_length, input_dim)
    # train_data = torch.randn(100, 30, input_dim)  # 100 sequences of 30 time steps

    # train_data = train_data.to(device)
    losses = list()
    model.train()
    # Training loop
    desc = desc_tqdm if desc_tqdm is not None else "TCN Autoencoder"
    pbar = tqdm(range(num_epochs), desc=desc)
    for _ in pbar:
        # pbar = tqdm(dataloader, desc=f"LSTM AutoEncoder Epoch {epoch+1}/{num_epochs}")
        runningLoss = 0
        n = 0
        for x in dataloader:
            optimizer.zero_grad()
            x = x.type(torch.float)[:, None, :].to(device)
            # print(x.shape)
            # Forward pass
            output = model(x)
            
            # Compute reconstruction loss
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            n += 1
        pbar.set_postfix(Loss=loss.item())
        losses.append(runningLoss/n)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return model, losses

def test_tcn_autoencoder(train_samples, correct_samples, incorrect_samples, model, use_gpu=True, batch_size=1000, num_workers=0):
    dataset_correct = GazeBasedInteraction(correct_samples)
    # dataloader_correct = DataLoader(dataset_correct, batch_size=batch_size, shuffle=False, num_wokers=num_workers)
    dataset_incorrect = GazeBasedInteraction(incorrect_samples)
    # dataloader_incorrect = DataLoader(dataset_incorrect, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataset_train = GazeBasedInteraction(train_samples)
    dataloader = {
        "Train": DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "Correct": DataLoader(dataset_correct, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "Incorrect": DataLoader(dataset_incorrect, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    device = torch.device("cuda") if torch.cuda.is_available() and use_gpu else torch.device("cpu")
    model = model.to(device)
    model.eval()
    mses = dict()
    with torch.no_grad():
        for k in ["Train", "Correct", "Incorrect"]:
            mses[k] = list()
            for x in dataloader[k]:
                x = x.type(torch.float)[:, None, :].to(device)
                out = model(x)
                mse = torch.mean((out - x)**2, dim=(1, 2))
                mses[k].append(mse)
            mses[k] = torch.concat(mses[k])
    return mses["Train"], mses["Correct"], mses["Incorrect"]


def main():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    # correct_data = np.load("Data/Dataset_Prepare/angles_gaze_Correct_f83_b500_a200.npy")
    # incorrect_data = np.load("Data/Dataset_Prepare/angles_gaze_Incorrect_f83_b500_a200.npy")
    f = 62
    a = 200
    b = 500
    fps = 90

    cond = "headAndGaze"
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

    models = [TCN_VAE(train_data.shape[1], 1, [8, 16, 32], 3, 16),
              LSTM_VAE(train_data.shape[1], 1, 40, 10, 1), 
              LSTMAE(1, 40, 10, 1), 
              LSTMAE_small(1, 40, 1),
              TCNAE(train_data.shape[1], 1, [8, 16, 32], 3, 16)]
              
    for m in models:
        name = m.__class__.__name__
        if isinstance(m, (LSTM_VAE, TCN_VAE)):
            criterion = vae_loss
        else:
            criterion = nn.MSELoss()
        model, losses = train_autoencoder(m, train_data, 1000, 200, criterion, 1e-4, desc_tqdm=name)
        # model, losses = train_lstm_autoencoder(train_data, batch_size=1000, num_epochs=400, lstm_model=m, num_workers=0, use_gpu=True,
        #                                        lstm_hidden_dim=128, lstm_latent_dim=32, lstm_num_layers=1, learning_rate=1e-3)
        print(f"{name} Count of parameters:", count_parameters(model))
        # plt.plot(losses)
        # plt.title(f"Train losses {name}")
        # plt.show()
        # mse_train, mse_correct, mse_incorrect = test_lstm_autoencoder(train_data, test_correct, test_incorrect, model)
        mse_train, mse_correct, mse_incorrect = test_autoencoder(train_data, test_correct, test_incorrect, model)
        th = np.percentile(mse_train.cpu().numpy(), 95)
        correct_acc = np.mean((mse_correct < th).cpu().numpy())
        incorrect_acc = np.mean((mse_incorrect > th).cpu().numpy())
        print(f"{name=}; {correct_acc=:.3f}; {incorrect_acc=:.3f}")
        fig, (ax_loss, ax_hist) = plt.subplots(2, figsize=(10,5))
        ax_loss.plot(losses)
        ax_loss.set_title(f"Train loss")
        sns.histplot({"Correct": mse_correct.cpu().numpy(), "Incorrect": mse_incorrect.cpu().numpy()},
                     multiple="layer", common_norm=False, stat="percent", ax=ax_hist)
        ax_hist.axvline(th, color="red", linestyle="--", label="Threshold")
        ax_hist.set_title("MSE Histogram")
        fig.suptitle(f"{name}; {cond}; {correct_acc=:.3f}; {incorrect_acc=:.3f}")
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
    plt.show()

def main_lstm():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    # correct_data = np.load("Data/Dataset_Prepare/angles_gaze_Correct_f83_b500_a200.npy")
    # incorrect_data = np.load("Data/Dataset_Prepare/angles_gaze_Incorrect_f83_b500_a200.npy")
    f = 83
    a = 200
    b = 500

    cond = "nod"
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
    # n = int(len(train_data) * 0.7)
    for name, m in [("ChatGPT", LSTMAutoencoder), ("GIT", LSTMAE)]:
        model, losses = train_lstm_autoencoder(train_data, batch_size=1000, num_epochs=400, lstm_model=m, num_workers=0, use_gpu=True,
                                               lstm_hidden_dim=128, lstm_latent_dim=32, lstm_num_layers=1, learning_rate=1e-3)
        print(f"{name} Count of parameters:", count_parameters(model))
        # plt.plot(losses)
        # plt.title(f"Train losses {name}")
        # plt.show()
        mse_train, mse_correct, mse_incorrect = test_lstm_autoencoder(train_data, test_correct, test_incorrect, model)
        th = np.percentile(mse_train.cpu().numpy(), 95)
        correct_acc = np.mean((mse_correct < th).cpu().numpy())
        incorrect_acc = np.mean((mse_incorrect > th).cpu().numpy())
        print(f"{name=}; {correct_acc=:.3f}; {incorrect_acc=:.3f}")
        fig, (ax_loss, ax_hist) = plt.subplots(2, figsize=(10,5))
        ax_loss.plot(losses)
        ax_loss.set_title(f"Train loss")
        sns.histplot({"Correct": mse_correct.cpu().numpy(), "Incorrect": mse_incorrect.cpu().numpy()},
                     multiple="layer", common_norm=False, stat="percent", ax=ax_hist)
        ax_hist.axvline(th, color="red", linestyle="--", label="Threshold")
        ax_hist.set_title("MSE Histogram")
        fig.suptitle(f"{name}; {cond}; {correct_acc=:.3f}; {incorrect_acc=:.3f}")
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
    plt.show()

def main_tcn():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    # correct_data = np.load("Data/Dataset_Prepare/angles_gaze_Correct_f83_b500_a200.npy")
    # incorrect_data = np.load("Data/Dataset_Prepare/angles_gaze_Incorrect_f83_b500_a200.npy")
    f = 83
    a = 200
    b = 500

    cond = "nod"
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
    # n = int(len(train_data) * 0.7)
    model, losses = train_tcn_autoencoder(train_data)
    print(f"Count of parameters:", count_parameters(model))
    # plt.plot(losses)
    # plt.title(f"Train losses {name}")
    # plt.show()
    mse_train, mse_correct, mse_incorrect = test_tcn_autoencoder(train_data, test_correct, test_incorrect, model)
    th = np.percentile(mse_train.cpu().numpy(), 95)
    correct_acc = np.mean((mse_correct < th).cpu().numpy())
    incorrect_acc = np.mean((mse_incorrect > th).cpu().numpy())
    print(f"TCN {correct_acc=:.3f}; {incorrect_acc=:.3f}")
    fig, (ax_loss, ax_hist) = plt.subplots(2, figsize=(10,5))
    ax_loss.plot(losses)
    ax_loss.set_title(f"Train loss")
    sns.histplot({"Correct": mse_correct.cpu().numpy(), "Incorrect": mse_incorrect.cpu().numpy()},
                    multiple="layer", common_norm=False, stat="percent", ax=ax_hist)
    ax_hist.axvline(th, color="red", linestyle="--", label="Threshold")
    ax_hist.set_title("MSE Histogram")
    fig.suptitle(f"TCN; {cond}; {correct_acc=:.3f}; {incorrect_acc=:.3f}")
    fig.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.001)
    plt.show()

if __name__=="__main__":
    main()