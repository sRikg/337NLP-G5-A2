import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    """
    LSTM model in encoder-decoder structure.
    """

    def __init__(self, num_layers, num_tokens, dim_emb, num_hiddens):
        super(LSTM, self).__init__()
        self.num_tokens = num_tokens
        self.dropout = nn.Dropout(0.3)
        self.encoder = nn.Embedding(num_tokens, dim_emb)
        self.lstm = nn.LSTM(dim_emb, num_hiddens, num_layers, dropout=0.3)
        self.decoder = nn.Linear(num_hiddens, num_tokens)
        self.num_layers = num_layers
        self.num_hidden = num_hiddens
        # Tie weights
        self.decoder.weight = self.encoder.weight
        self.init_weights()

    def forward(self, input, hidden):
        x = self.encoder(input)
        x = self.dropout(x)
        x, h = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.decoder(x)
        x = x.view(-1, self.num_tokens)

        return F.log_softmax(x, dim=1), h

    def init_weights(self):
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())

        return weight.new_zeros(self.num_layers, batch_size, self.num_hidden), \
               weight.new_zeros(self.num_layers, batch_size, self.num_hidden)
