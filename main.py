import torch.nn as nn
from deeplib import training
from torch.utils.data import Dataset
from preprocessing import Preprocess
from torch.autograd import Variable
import torch


class ToxicityDataset(Dataset):
    def __init__(self, input, target):
        super(ToxicityDataset, self).__init__()

        self.input = input
        self.target = target
        if len(input) != len(target):
            raise Exception('Input and target are not the same length!')

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        return (self.input[item], self.target[item])


class GRU(nn.Module):
    def __init__(self, vocabulary_size):
        super(GRU, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        #self.lstm = nn.LSTM(vocabulary_size, 20, bidirectional=True, batch_first=True)
        #self.lstm_hidden = None
        self.gru = nn.GRU(vocabulary_size, 20, bidirectional=True, batch_first=True)
        self.gru_hidden = None
        self.lc = nn.Linear(40, 2)

    def forward(self, x):
        #self.lstm_hidden = (Variable(torch.randn(2, 1, 40)).double(), Variable(torch.randn(2, 1, 40)).double())
        #self.gru_hidden = Variable(torch.randn(2, 1, 40)).double()
        #x, self.lstm_hidden = self.lstm(x, None)
        x, self.gru_hidden = self.gru(x, None)
        x = x[:, -1, :].squeeze(1)
        x = self.dropout(x)
        output = self.lc(x)
        return output


if __name__ == '__main__':
    pre = Preprocess()
    pre.build_vectors()
    use_GPU = True
    dataset = ToxicityDataset(pre.vectors, pre.targets)
    # Without sentiment
    #gru = GRU(360).double()
    # With sentiment
    gru = GRU(373).double()
    if use_GPU:
        gru.cuda()
    training.train(gru, dataset, 20, 1, 0.1, use_gpu=use_GPU)

