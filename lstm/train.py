import torch
import torch.nn as nn
import argparse
import math

import model
import data


# Set arguments for the model training.
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../wiki.')
parser.add_argument('--emb', type=int, default=100)
parser.add_argument('--num_hidden', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.1)
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

def create_batch(data, batch_size):
    num_batch = data.size(0) // batch_size
    data = data.narrow(0, 0, num_batch*batch_size)
    data = data.view(batch_size, -1).t().contiguous()

    return data.to(device)

def get_batch(source, i):
    seq_len = min(args.window_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+seq_len+1].view(-1)

    return data, target

def clear_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(clear_hidden(v) for v in h)


corpus = data.Corpus(args.data)

train_data = create_batch(corpus.train, args.batch_size)
val_data = create_batch(corpus.val, args.batch_size)
test_data = create_batch(corpus.test, args.batch_size)

# Model definition.
num_tokens = len(corpus.encoding.index2word)
model = model.LSTM(args.num_layers, num_tokens, args.emb, args.num_hidden)
criterion = nn.NLLLoss()

# Train
def train():
    model.train()
    total_loss = 0

    hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.window_size)):
        data, targets = get_batch(train_data, i)
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

# Eval or test.
def evaluate(data):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(args.batch_size)

    with torch.no_grad():
        for i in range(0, data.size(0)-1, args.window_size):
            x, targets = get_batch(data, i)
            x, hidden = model(x, hidden)
            hidden = clear_hidden(hidden)
            total_loss += criterion(x, targets).item() * len(x)

    return total_loss / (len(data) - 1)


# Training process.
for epoch in range(1, args.epochs + 1):
    train()

    # Eval
    val_loss = evaluate(val_data)
    print('epoch {}, val loss {}, val perplexity {}'.format(epoch, val_loss, math.exp(val_loss)))

    # Testing
    test_loss = evaluate(test_data)
    print('epoch {}, test loss {}, test perplexity {}'.format(epoch, test_loss, math.exp(test_loss)))
