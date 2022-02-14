import torch
import torch.nn as nn
import argparse
import math
import model
import data
import matplotlib.pyplot as plt
import time


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

# print('# of train samples:', num_sample_train)
# print('# of val samples:', num_sample_val)
# print('# of test samples:', num_sample_test)

"""
# Preview data
data, targets = get_batch(train_data, 0)
print('size of data and target:', data.shape,targets.shape)  # torch.Size([30,20]) torch.Size([600])
# print('view data', data[1:10,:])
# print('view target', targets[:200])
"""


# Model definition.
num_tokens = len(corpus.encoding.index2word)
model = model.RNN(args.num_layers, num_tokens, args.emb, args.batch_size, device).to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train
def train():
    model.train()
    total_loss = 0
    # actually time steps, not batch
    for batch, i in enumerate(range(0, train_data.size(1) - 1, args.window_size)):
        data, targets = get_batch(train_data, i)
        data = data.to(device)
        targets = targets.to(device)
        model.zero_grad()

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
            avg_loss = total_loss / (batch + 1)
            print('epoch {}, perplexity {}'.format(epoch, round(math.exp(avg_loss), 2)))

    perplex = math.exp(total_loss / (train_data.size(1) // args.window_size))
    return round(perplex, 2)


# Eval or test.
def evaluate(data, test=False):
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for i in range(0, data.size(1)-1, args.window_size):
            x, targets = get_batch(data, i)
            output = model(x)
            total_loss += criterion(output, targets).item()
    perplex = math.exp(total_loss / (data.size(1) // args.window_size))

    return round(perplex, 2)


# Training process.
train_perplex = []
eval_perplex = []
test_perplex = []
start_time = time.time()

for epoch in range(1, args.epochs + 1):
    # train
    train_perplex.append(train())
    print('epoch {}, train perplexity {}'.format(epoch, train_perplex[-1]))

    # Eval
    eval_perplex.append(evaluate(val_data))
    print('epoch {}, val perplexity {}'.format(epoch, eval_perplex[-1]))

    # Testing
    test_perplex.append(evaluate(test_data, test=True))
    print('epoch {}, test perplexity {}'.format(epoch, test_perplex[-1]))

elapsed = (time.time() - start_time)/60
print(f'Total Running Time: {elapsed:.2f} min')

# Plot perplexities
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(range(1, args.epochs+1), train_perplex, label='Training')
ax.plot(range(1, args.epochs+1), eval_perplex, label='Validation')
ax.plot(range(1, args.epochs+1), test_perplex, label='Test')
ax.legend(loc='upper right')
plt.ylabel('Perplexity')
plt.xlabel('Epoch')
plt.show()
fig.savefig(f'Figures/perplexity_lr{args.lr}.png')

# final perplexity
print(f'final perplexities: train - {train_perplex[-1]}, eval - {eval_perplex[-1]},'
      f' test - {test_perplex[-1]}')
