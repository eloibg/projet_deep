import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from deeplib import training
from torch.utils.data import Dataset, DataLoader
from preprocessing import Preprocess


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

        self.gru = nn.GRU(vocabulary_size, 20, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.lc = nn.Linear(40, 2)

    def forward(self, x, hidden):
        input = x
        output1, h_n = self.gru(input, hidden)
        x = self.dropout(output1[:, -1, :].squeeze(1))
        output2 = self.lc(x)
        return output2, h_n


if __name__ == '__main__':
    pre = Preprocess()
    pre.build_vectors()
    dataset = ToxicityDataset(pre.vectors, pre.y_train)

    train_loader = DataLoader(dataset, batch_size=1)
    gru = GRU(372).double()

    training.train(gru, dataset, 10, 1, 0.1)

