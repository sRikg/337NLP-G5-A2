import torch

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
