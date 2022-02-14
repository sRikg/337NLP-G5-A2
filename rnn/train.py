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
print('device:', device)


# Create a batch from the given data.
def create_batch(data, batch_size):
    num_batch = data.size(0) // batch_size
    # num_sample = num_batch*batch_size
    data = data.narrow(0, 0, num_batch*batch_size)
    data = data.view(batch_size, -1).contiguous()   # --modified-- #

    return data.to(device)


# Provide a batch within the source.
def get_batch(source, i):
    seq_len = min(args.window_size, len(source[0]) - 1 - i)
    data = source[:, i:i+seq_len]
    target = source[:, i+1:i+seq_len+1].t().reshape(-1)

    # return data.to(device), target.to(device)
    return data, target


# Clear the hidden from gradient.
def clear_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(clear_hidden(v) for v in h)


corpus = data.Corpus(args.data)

train_data = create_batch(corpus.train, args.batch_size)
val_data = create_batch(corpus.val, args.batch_size)
test_data = create_batch(corpus.test, args.batch_size)

num_sample_train = train_data.shape[0]*train_data.shape[1]
num_sample_val = val_data.shape[0]*val_data.shape[1]
num_sample_test = test_data.shape[0]*test_data.shape[1]


# Preview data
data, targets = get_batch(train_data, 0)
print('size of data and target:', data.shape,targets.shape)  # torch.Size([30,20]) torch.Size([600])
# print('view data', data[1:10,:])
# print('view target', targets[:200])



# Model definition.
num_tokens = len(corpus.encoding.index2word)
model = model.RNN(args.num_layers, num_tokens, args.emb, args.batch_size).to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train
def train():
    model.train()
    total_loss = 0

    # hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.window_size)):
        data, targets = get_batch(train_data, i)
        model.zero_grad()

        # hidden = clear_hidden(hidden)
        print('model on CUDA?', next(model.parameters()).is_cuda)
        x = model(data)
        loss = criterion(x, targets)
        loss.backward()
        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # Grad clip
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-args.lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            tmp_loss = total_loss / args.log_interval
            print('epoch {}, loss {}, perplexity {}'.format(epoch, tmp_loss, math.exp(tmp_loss)))
            total_loss = 0
    perplex = math.exp(total_loss / num_sample_train)
    return perplex


# Eval or test.
def evaluate(data, test = False):
    model.eval()
    total_loss = 0.
    # hidden = model.init_hidden(args.batch_size)

    with torch.no_grad():
        for i in range(0, data.size(0)-1, args.window_size):
            x, targets = get_batch(data, i)
            output = model(x)
            # hidden = clear_hidden(hidden)
            total_loss += criterion(output, targets).item()
    if not test:
        perplex = math.exp(total_loss / num_sample_val)
    else:
        perplex = math.exp(total_loss / num_sample_test)

    return perplex


# Training process.
train_perplex = []
eval_perplex = []
test_perplex = []

for epoch in range(1, args.epochs + 1):
    # train
    train_perplex.append(train())

    # Eval
    eval_perplex.append(evaluate(val_data))
    print('epoch {}, val perplexity {}'.format(epoch, eval_perplex[-1]))

    # Testing
    test_perplex.append(evaluate(test_data, test=True))
    print('epoch {}, test perplexity {}'.format(epoch, test_perplex))

# final perplexity
print(f'final perplexities: train - {train_perplex[-1]}, eval - {eval_perplex[-1]},'
      f' test - {test_perplex[-1]}')
