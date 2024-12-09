import torch
from torch import nn
import torch.nn.functional as F
from TCN import TCNBlock


def vae_loss(reconstructed, original, mu, log_var, beta=0.1):
    reconstruction_loss = F.mse_loss(reconstructed, original, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + beta * kl_loss

class LSTM_VAE(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_size, latent_dim, num_layers):
        super(LSTM_VAE, self).__init__()
        self.seq_len = seq_len
        self.lstm_enc = nn.LSTM(input_size=input_dim,
                                hidden_size=hidden_size,
                                num_layers=1,
                                batch_first=True)
        
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_size)

        self.lstm_dec = nn.LSTM(input_size=input_dim,
                                hidden_size=hidden_size,
                                num_layers=1,
                                batch_first=True)
        self.fc = nn.Linear(hidden_size, input_dim)
    
    def encoder(self, x):
        out, (h_state, c_state) = self.lstm_enc(x)
        x_enc = h_state.squeeze(0)
        # x_enc = x_enc.unsqueeze(1).repeat(1, x.shape[1], 1)
        mu = self.fc_mu(x_enc)
        log_var = self.fc_logvar(x_enc)
        return mu, log_var

    def decoder(self, z):
        hidden = self.decoder_input(z).unsqueeze(0)
        repeated_input = torch.zeros((z.size(0), self.seq_len, 1), device=z.device)
        dec_out, _ = self.lstm_dec(repeated_input, (hidden, torch.zeros_like(hidden)))
        dec_out = self.fc(dec_out)
        return dec_out
        
    def reparameterize(self, mu, log_var):
        # log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        x_enc = self.reparameterize(mu, log_var)
        x_dec = self.decoder(x_enc)
        return x_dec, mu, log_var


class TCN_VAE(nn.Module):
    def __init__(self, seq_len, input_dim, num_channels, kernel_size, latent_dim):
        super(TCN_VAE, self).__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.encoder_tcn = nn.Sequential(
            *[TCNBlock(input_dim if i == 0 else num_channels[i-1],
                       num_channels[i], kernel_size, dilation=2**i)
              for i in range(len(num_channels))]
        )
        
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(seq_len * num_channels[-1] * input_dim, latent_dim)
        self.fc_logvar = nn.Linear(seq_len * num_channels[-1] * input_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, seq_len * num_channels[-1] * input_dim)

        self.decoder_tcn = nn.Sequential(
            *[TCNBlock(num_channels[-i-1],
                       num_channels[-i-2] if i < len(num_channels)-1 else input_dim,
                       kernel_size, dilation=2**i)
              for i in range(len(num_channels))]
        )
    
    def encoder(self, x):
        x = self.encoder_tcn(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, log_var):
        # log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decoder(self, z):
        z = self.fc_dec(z)
        z = z.view(z.size(0), self.num_channels[-1], self.seq_len)
        reconstructed = self.decoder_tcn(z)
        return reconstructed
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


if __name__=="__main__":
    import torch
    tensor = torch.randn(5, 1, 63)
    parameters = {
        'input_dim': 1,
        'hidden_size': 20,
        'latent_dim': 10,
        'num_layers': 3
    }
    # model = LSTM_VAE(1, 20, 10, 3)
    model = TCN_VAE(63, 1, [8, 16, 32], 3, 16)
    tensor_rec, mu, log_var = model(tensor)
    print(f"{tensor.shape=}\n{tensor_rec.shape}\n{mu.shape=}\n{log_var.shape=}")
    print(isinstance(model, LSTM_VAE))
    print(model.__class__.__name__)
    