from pathlib import Path
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST

DATA_ROOT = Path("./data/")

RESULTS_ROOT = Path('./results')
DATA_TRAIN_INDICES = Path('./train_indices.pkl')

DS_PATH_CFG = {
    'cifar10_train':
        (CIFAR10, {'root': DATA_ROOT / 'cifar10', 'train': True, 'download': True}),
    'cifar10_test':
        (CIFAR10, {'root': DATA_ROOT / 'cifar10', 'train': False, 'download': True}),
    'cifar100_train':
        (CIFAR100, {'root': DATA_ROOT / 'cifar100', 'train': True, 'download': True}),
    'cifar100_test':
        (CIFAR100, {'root': DATA_ROOT / 'cifar100', 'train': False, 'download': True}),
}


DS_SPLIT_CFG = {
    'cifar10_train': [
        100, 
        200, 
        300, 
        400, 
        500, 
        1000, 
        2500, 
        5000, 
        6000, 
        7000, 
        8000, 
        9000,
        10000,
        11000,
        12000,
        13000,
        14000,
        15000, 
        16000, 
        17000, 
        18000, 
        19000, 
        20000, 
        25000, 
        30000, 
        35000, 
        40000, 
        45000
    ]
}
