import math
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate

def pad_collate(batch):
    batch.sort(key=lambda list: len(list[0]), reverse=True)
    len_list = np.zeros(len(batch), dtype=int)
    for i, item in enumerate(batch):
        len_list[i] = int(len(item[0]))
    max_len = len_list[0]
    for i, item in enumerate(batch):
        batch[i] = (np.concatenate((item[0], np.zeros((int(max_len-len(item[0])), item[0].shape[1])))), batch[i][1])
    return default_collate(batch), len_list


def train_valid_loaders(dataset, batch_size, train_split=0.75, shuffle=True):
    num_data = len(dataset)
    indices = np.arange(num_data)

    if shuffle:
        np.random.shuffle(indices)

    split = math.floor(train_split * num_data)
    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, collate_fn=pad_collate)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=valid_sampler, collate_fn=pad_collate)

    return train_loader, valid_loader