from torchvision.transforms import ToTensor
from deeplib.data import train_valid_loaders
from deeplib.training import validate


def test(model, dataset, batch_size, use_gpu=True):
    loader, _ = train_valid_loaders(dataset, batch_size)
    return validate(model, loader, use_gpu)
