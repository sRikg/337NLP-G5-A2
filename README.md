# Next-word prediction on Wiki text

This repo includes two language models implemented on the wiki text dataset: RNN and LSTM

## Hyperparameters

- Num hidden layers -> 2

- Embedding dimension -> 100

- Time step -> 30

- Num batches -> 20

- Num Epochs -> 20


## RNN
RNN is implemented from basic PyTorch components (such as linear layer and tanh activation function) within a for loop that iterate through each time step. The model takes in a 20x30 matrix of word indices as input which is then used to look up the embeddings for each word. A for loop is then used to iterate through each column of this matrix to represent time steps. In the first loop, the embeddings of the first words in every of the 20 batches are fed into the first linear layer to generate hidden layer 1. Then hidden layer 1 is fed into the second linear layer to generate hidden layer 2. Following this, a hidden layer 2 is multiplied by the transpose of the embedding matrix to generate logits which are run through a log softmax layer to provide predictions.

To run the model, run `RNN/train.py`. A list of hyperparameters can be passed in as arguments. See the file for details

## LSTM
LSTM is implemented in a similar manner utilizing LSTM cell provided in PyTorch. Dropout and gradient clipping are applied to avoid gradient explosion and reduce overfitting.

To run the model, run `LSTM.py`. A list of hyperparameters can be passed in as arguments. See the file for details
