'''Main function to demo covid classification'''

# Necessary packages
import os 
import re
import PIL
import torch
import timm

from torchvision import transforms, models, datasets
from randaugment import RandAugment, ImageNetPolicy, Cutout
# My packages

from data_helper import data_dir
from training import create_model

def main(args):
    ''' Main function for 
    
    Args:
        - from_id: start index to file list
        
    Returns:
        - Write file missing data
        - Write file imputed values 
    
    '''

    # Parameters
    batch_size = args.batch_size 
    flag_model = args.flag_model
    # Define your transforms for the training and testing sets
    data_transforms = {
        'train': transforms.Compose([
                    transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    RandAugment(),
                    ImageNetPolicy(),
                    Cutout(size=16),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])
                ]),
        'test': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])
                ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    data_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffe=True, num_workers=4, pin_memory=True) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    # we get the class_to_index in the data_Set but what we really need is the cat_to_names  so we will create
    _ = image_datasets['train'].class_to_idx
    cat_to_name = {_[i]: i for i in list(_.keys())}

    # Run this to test the data loader
    images, labels = next(iter(data_loader['test']))
    
    # Create pre_trained model
    model = create_model(flag_model=flag_model, flag_pretrained=True)

