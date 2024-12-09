from torch import nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder LSTM
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        # Encode
        _, (hidden, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(hidden[-1])
        
        # Decode
        hidden = self.decoder_fc(latent).unsqueeze(0)
        output, _ = self.decoder_lstm(hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2))
        
        return output
    
    import torch.nn as nn

#%% stolen from https://github.com/matanle51/LSTM_AutoEncoder/blob/master/models/LSTMAE.py

# Encoder Class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):#, seq_len):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        # self.seq_len = seq_len

        self.lstm_enc = nn.LSTM(input_size=input_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers,
                                dropout=dropout, batch_first=True)

    def forward(self, x):
        out, (last_h_state, last_c_state) = self.lstm_enc(x)
        x_enc = last_h_state[-1].squeeze(dim=0)
        x_enc = x_enc.unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_enc, out


# Decoder Class
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout): # , seq_len, use_act):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        # self.seq_len = seq_len
        # self.use_act = use_act  # Parameter to control the last sigmoid activation - depends on the normalization used.
        # self.act = nn.Sigmoid()

        self.lstm_dec = nn.LSTM(input_size=hidden_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers,
                                dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, z):
        # z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, (hidden_state, cell_state) = self.lstm_dec(z)
        dec_out = self.fc(dec_out)
        # if self.use_act:
        #     dec_out = self.act(dec_out)

        return dec_out, hidden_state


# LSTM Auto-Encoder Class
class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim, num_layers, dropout_ratio=0): #, seq_len, use_act=False):
        super(LSTMAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        # self.seq_len = seq_len

        self.encoder = Encoder(input_size=input_dim, hidden_size=hidden_size, dropout=dropout_ratio, num_layers=num_layers) # , seq_len=seq_len)
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_size)
        self.decoder = Decoder(input_size=input_dim, hidden_size=hidden_size, dropout=dropout_ratio, num_layers=num_layers) # , seq_len=seq_len, use_act=use_act)

    def forward(self, x, return_last_h=False, return_enc_out=False):
        x_enc, enc_out = self.encoder(x)
        x_latent = self.encoder_fc(x_enc)
        x_latent = self.decoder_fc(x_latent)
        x_dec, last_h = self.decoder(x_latent)

        if return_last_h:
            return x_dec, last_h
        elif return_enc_out:
            return x_dec, enc_out
        return x_dec
    

class LSTMAE_small(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout_ratio=0): #, seq_len, use_act=False):
        super(LSTMAE_small, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio

        self.encoder = Encoder(input_size=input_dim, hidden_size=hidden_size, dropout=dropout_ratio, num_layers=num_layers) # , seq_len=seq_len)
        self.decoder = Decoder(input_size=input_dim, hidden_size=hidden_size, dropout=dropout_ratio, num_layers=num_layers) # , seq_len=seq_len, use_act=use_act)

    def forward(self, x):
        x_enc, _ = self.encoder(x)
        x_dec, _ = self.decoder(x_enc)
        return x_dec

if __name__=="__main__":
    import torch
    tensor = torch.randn(5, 83,1)
    model = LSTMAE_small(1, 20, 3)
    tensor_rec = model(tensor)
    print(f"{tensor.shape=}\n{tensor_rec.shape=} ")
    loss = nn.MSELoss()(tensor_rec, tensor)
    print(loss)