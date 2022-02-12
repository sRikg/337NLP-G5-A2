import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    """
    LSTM model in encoder-decoder structure.
    """

    def __init__(self, num_layers, num_tokens, dim_emb, num_hiddens):
        super(RNN, self).__init__()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_hidden = num_hiddens
        self.embeddings = nn.Embedding(num_tokens, dim_emb)
        self.linear1 = nn.Linear(3100, 200)
        self.linear2 = nn.Linear(200, 100)
        # Tie weights
        self.init_weights()

        # self.dropout = nn.Dropout(0.3)
        # self.lstm = nn.LSTM(dim_emb, num_hiddens, num_layers, dropout=0.3)
        # self.decoder = nn.Linear(num_hiddens, num_tokens)
        # self.decoder.weight = self.encoder.weight

    def forward(self, input, hidden):
        h1 = self.linear1(input)
        h2 = self.linear2(h1)
        logits = torch.matmul(h2, self.embeddings.weight)
        # x = self.dropout(x)
        # x = x.view(-1, self.num_tokens)

        return F.log_softmax(logits), h1, h2

    def init_weights(self):
        nn.init.uniform_(self.linear1.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear2.weight, -0.1, 0.1)

    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters())
    #
    #     return weight.new_zeros(self.num_layers, batch_size, self.num_hidden), \
    #            weight.new_zeros(self.num_layers, batch_size, self.num_hidden)
