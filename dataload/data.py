from os.path import join
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop
from dataload.dataset import DatasetFromFolderEval, DatasetFromFolder
# from data_transform import *

def transform():
    return Compose([
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def get_training_set(data_dir, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, 'GT')
    lr_dir = join(data_dir, 'LR')
    return DatasetFromFolder(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation, transform=transform())

def get_eval_set(data_dir, upscale_factor):
    hr_dir = join(data_dir, 'GT')
    lr_dir = join(data_dir, 'LR')
    return DatasetFromFolderEval(hr_dir, lr_dir, upscale_factor, transform=transform())

