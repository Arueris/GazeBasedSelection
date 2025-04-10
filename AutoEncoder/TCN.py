import torch
from torch import nn

#%% from ChatGPT created

# Define a single TCN block
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=(kernel_size - 1) * dilation // 2,  # Compute padding to keep sequence length constant
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class TCNAE(nn.Module):
    def __init__(self, seq_len, input_dim, num_channels, kernel_size=3, latent_dim=16):
        super(TCNAE, self).__init__()
        self.ts_len = seq_len
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        # Encoder: Stack of TCN blocks
        self.encoder = nn.Sequential(
            *[TCNBlock(input_dim if i == 0 else num_channels[i-1],
                       num_channels[i], kernel_size, dilation=2**i) 
              for i in range(len(num_channels))]
        )
        
        # Bottleneck: Fully connected layer
        self.flatten = nn.Flatten()  # Flatten time dimension for bottleneck
        self.fc_enc = nn.Linear(seq_len * num_channels[-1] * input_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, seq_len * num_channels[-1] * input_dim)
        
        # Decoder: Reverse TCN blocks
        self.decoder = nn.Sequential(
            *[TCNBlock(num_channels[-i-1],
                       num_channels[-i-2] if i < len(num_channels)-1 else input_dim,
                       kernel_size, dilation=2**i)
              for i in range(len(num_channels))]
        )
    
    def __repr__(self):
        return f"TCNAE: ts_len {self.ts_len}, input_dim {self.input_dim}, num_channels {self.num_channels}, kernel_size {self.kernel_size}, latent_dim {self.latent_dim}"

    def forward(self, x):

        x = x.reshape(x.size(0), 1, x.size(-1))

        # Encoder
        encoded = self.encoder(x)
        
        # Bottleneck
        encoded_flat = self.flatten(encoded)
        latent = self.fc_enc(encoded_flat)
        reconstructed_flat = self.fc_dec(latent)
        
        # Reshape back to sequence format
        reconstructed = reconstructed_flat.view(encoded.size())
        
        # Decoder
        reconstructed = self.decoder(reconstructed)
        return reconstructed
    

if __name__=="__main__":
    parameter = {
        'seq_len': 45,
        'num_channels': [16, 32, 64],
        'input_dim': 1,
        'kernel_size': 5,
        'latent_dim': 16
    }
    model = TCNAE(**parameter)
    # model = TCNAE(83, 1, [16, 32, 64])
    tensor = torch.randn(5,45)
    tensor = tensor[:, None, :]
    res = model(tensor)
    res_encoder = model.encoder(tensor)
    print(f"{tensor.shape=}")
    print(f"{res.shape=}")
    print(f"{res_encoder.shape=}")
    print(model)
