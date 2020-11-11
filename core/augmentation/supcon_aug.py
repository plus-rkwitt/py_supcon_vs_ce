"""
Augmentation policy taken from 
https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
which is the implementation of 

@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
"""

import torchvision.transforms as transforms


supcon_aug = \
    transforms.Compose([
        transforms.RandomResizedCrop(
            size=32, # This parameter is setable in the original implementation
            # and hard-coded to its default value in our implementation. 
            scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])