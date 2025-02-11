from config import *
from helper import *
from model import *

from tqdm import tqdm
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Set device and random seed for reproducibility.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DATA_DIR = "/home/sahme175/data"
    
    mnist_data = datasets.MNIST(DATA_DIR, download=False, train=True,
                                transform=transforms.ToTensor())

    mnist_test = datasets.MNIST(DATA_DIR, download=False, train=False,
                                transform=transforms.ToTensor())

    # Split dataset into training and validation and create data loaders
    batch_size = 128
    train_dataset, val_dataset = torch.utils.data.random_split(mnist_data, [int(0.8 * len(mnist_data)), len(mnist_data) - int(0.8 * len(mnist_data))])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=10000, shuffle=False, pin_memory=True, num_workers=4)

    
    best_config, best_score, best_model = grid_search_nas(train_loader, val_loader, device, epochs=10, )