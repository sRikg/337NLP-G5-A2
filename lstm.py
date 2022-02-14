import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math


# Set arguments for the model training.
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='wiki.')
parser.add_argument('--emb', type=int, default=100)
parser.add_argument('--num_hidden', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=10)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--window_size', type=int, default=30)
parser.add_argument('--seed', type=int, default=337)
parser.add_argument('--log_interval', type=int, default=300)

args = parser.parse_args()

# Set the random seed.
torch.manual_seed(args.seed)

# Set the device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Corpus():
    """ Transform words to index then to tensors."""
    def __init__(self, path):
        self.encoding = Encoding()
        self.train = self.process(path + 'train.txt')
        self.val = self.process(path + 'valid.txt')
        self.test = self.process(path + 'test.txt')

    def process(self, path):
        with open(path, 'r') as f:
            all = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for w in words:
                    ids.append(self.encoding.encode(w))
                all.append(torch.tensor(ids).type(torch.int64))
            res = torch.cat(all)

        return res


class Encoding():
    """ Encode word to index"""
    def __init__(self):
        self.word2index = {}
        self.index2word = []

    def encode(self, word):
        if word not in self.word2index:
            self.index2word.append(word)
            self.word2index[word] = len(self.index2word) - 1

        return self.word2index[word]


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


# Create a batch from the given data.
def create_batch(data, batch_size):
    num_batch = data.size(0) // batch_size
    data = data.narrow(0, 0, num_batch*batch_size)
    data = data.view(batch_size, -1).t().contiguous()

    return data.to(device)


# Provide a batch within the source.
def get_batch(source, i):
    seq_len = min(args.window_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+seq_len+1].view(-1)

    return data, target


# Clear the hidden from gradient.
def clear_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(clear_hidden(v) for v in h)


corpus = Corpus(args.data)

train_data = create_batch(corpus.train, args.batch_size)
val_data = create_batch(corpus.val, args.batch_size)
test_data = create_batch(corpus.test, args.batch_size)
# print('corpus.train.shape: {}'.format(corpus.train.shape))
# print('train_data.shape: {}'.format(train_data.shape))

# Model definition.
num_tokens = len(corpus.encoding.index2word)
model = LSTM(args.num_layers, num_tokens, args.emb, args.num_hidden).to(device)
criterion = nn.NLLLoss()

train_ppl_history = []
val_ppl_history = []
test_ppl_history = []


# Train
def train():
    model.train()
    total_loss = 0

    hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.window_size)):
        data, targets = get_batch(train_data, i)
        # print('data.shape: {}'.format(data.shape))
        # print('data: {}'.format(data))
        # print('targets.shape: {}'.format(targets.shape))
        model.zero_grad()

        hidden = clear_hidden(hidden)
        x, hidden = model(data, hidden)
        loss = criterion(x, targets)
        loss.backward()

        # Grad clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-args.lr)
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            tmp_loss = total_loss / args.log_interval
            print('epoch {}, loss {}, perplexity {}'.format(epoch, tmp_loss, math.exp(tmp_loss)))
            total_loss = 0

            train_ppl_history.append([epoch, batch, math.exp(tmp_loss)])


# Eval or test.
def evaluate(data):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(args.batch_size)

    with torch.no_grad():
        for i in range(0, data.size(0)-1, args.window_size):
            x, targets = get_batch(data, i)
            output, hidden = model(x, hidden)
            hidden = clear_hidden(hidden)
            total_loss += criterion(output, targets).item() * len(x)

    return total_loss / (len(data) - 1)


# Training process.
for epoch in range(1, args.epochs + 1):
    train()

    # Eval
    val_loss = evaluate(val_data)
    print('epoch {}, val loss {}, val perplexity {}'.format(epoch, val_loss, math.exp(val_loss)))
    val_ppl_history.append([epoch, math.exp(val_loss)])

    # Testing
    test_loss = evaluate(test_data)
    print('epoch {}, test loss {}, test perplexity {}'.format(epoch, test_loss, math.exp(test_loss)))
    test_ppl_history.append([epoch, math.exp(test_loss)])

print('Train perplexity history: {}'.format(train_ppl_history))
print('Val perplexity history: {}'.format(val_ppl_history))
print('Test perplexity history: {}'.format(test_ppl_history))