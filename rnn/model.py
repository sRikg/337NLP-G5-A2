import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    """
    RNN model implemented with basic PyTorch functions.
    """

    def __init__(self, num_layers, num_tokens, dim_emb, batch_size, device):
        super(RNN, self).__init__()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.embeddings = nn.Embedding(num_tokens, dim_emb)
        self.linear1 = nn.Linear(200, 100)
        self.linear2 = nn.Linear(200, 100)
        self.activation = nn.Tanh()
        self.h1 = torch.zeros(self.batch_size, 100).to(device)
        self.h2 = torch.zeros(self.batch_size, 100).to(device)
        # Tie weights
        self.init_weights()

        # self.num_hidden = num_hiddens
        # self.dropout = nn.Dropout(0.3)
        # self.lstm = nn.LSTM(dim_emb, num_hiddens, num_layers, dropout=0.3)
        # self.decoder = nn.Linear(num_hiddens, num_tokens)
        # self.decoder.weight = self.encoder.weight

    def forward(self, input):
        logits = torch.tensor(self.num_tokens, device=self.device)
        # Process a batch of 20 samples in each time step
        for col in range(input.shape[1]):
            curr_input = input[:, col].long().to(self.device)
            # print('current input on CUDA?', curr_input.is_cuda)
            x = self.embeddings(curr_input).to(self.device)    # should be size([20, 100])
            # print('current embed on CUDA?', x.is_cuda)
            # print('shape of x:', x.shape)
            # print('current h1 on CUDA?', self.h1.is_cuda)
            self.h1 = self.activation(self.linear1(torch.cat((x, self.h1), dim=1).to(self.device)))\
                .to(self.device)
            self.h2 = self.activation(self.linear2(torch.cat((self.h1, self.h2), dim=1).to(self.device)))\
                .to(self.device)   # size([20, 100])
            # print('shape of h2:', self.h2.shape)
            if col == 0:
                logits = torch.matmul(self.h2, self.embeddings.weight.t())
            else:
                logits = torch.vstack((logits, torch.matmul(self.h2, self.embeddings.weight.t())))  # size([20, num_tokens])
        # print('shape of logits:', logits.shape)  # size([600, num_tokens])
        # x = self.dropout(x)
        # x = x.view(-1, self.num_tokens)

        # Reset h1 & h2
        self.h1 = torch.zeros(self.batch_size, 100).to(self.device)
        self.h2 = torch.zeros(self.batch_size, 100).to(self.device)

        return F.log_softmax(logits, dim=1)

    def init_weights(self):
        nn.init.uniform_(self.linear1.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear2.weight, -0.1, 0.1)
