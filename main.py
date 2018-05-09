import torch.nn as nn
from deeplib import training
from torch.utils.data import Dataset
from preprocessing import Preprocess
import numpy as np


class ToxicityDataset(Dataset):
    def __init__(self, input, target):
        super(ToxicityDataset, self).__init__()

        self.input = input
        self.target = target
        if len(input) != len(target):
            raise Exception('Input and target are not the same length!')
        #self.pad()

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        return (self.input[item], self.target[item])

    def pad(self):
        max_len = 0
        for input in self.input:
            if input.shape[0] > max_len:
                max_len = input.shape[0]
        for i, input in enumerate(self.input):
            input_len = input.shape[0]
            input = np.concatenate((input, np.zeros((max_len-input_len, input.shape[1]))))
            self.input[i] = (input, input_len)

class GRU(nn.Module):
    def __init__(self, vocabulary_size):
        super(GRU, self).__init__()

        self.gru = nn.GRU(vocabulary_size, 20, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.lc = nn.Linear(40, 2)

    def forward(self, x, hidden):
        output1, h_n = self.gru(x, hidden)
        x = self.dropout(output1[:, -1, :].squeeze(1))
        output2 = self.lc(x)
        return output2, h_n


if __name__ == '__main__':
    pre = Preprocess()
    pre.build_vectors()
    use_GPU = True
    dataset = ToxicityDataset(pre.vectors, pre.y_train)
    gru = GRU(373).double()
    if use_GPU:
        gru.cuda()

    training.train(gru, dataset, 10, 1, 0.1, use_gpu=use_GPU)

