'''Main function to demo covid classification'''

# Necessary packages
import os 
import re
import PIL
import torch
import timm

from torchvision import transforms, models, datasets
from randaugment import RandAugment, ImageNetPolicy, Cutout
import torch.nn as nn
# My packages

from data_helper import data_dir, CHECK_POINT_PATH
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
    
    # Loaded pre_trained model
    model = create_model(flag_model=flag_model, flag_pretrained=True)

    # Create classifiers
    for param in model.parameters():
        param.requires_grad = True

    # Configs some last layers for my tasks classification covid
    fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 1000, bias=True)),
							     ('BN1', nn.BatchNorm2d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
								 ('dropout1', nn.Dropout(0.7)),
                                 ('fc2', nn.Linear(1000, 512)),
								 ('BN2', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
								 ('swish1', Swish()),
								 ('dropout2', nn.Dropout(0.5)),
								 ('fc3', nn.Linear(512, 128)),
								 ('BN3', nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
							     ('swish2', Swish()),
								 ('fc4', nn.Linear(128, 3)),
								 ('output', nn.Softmax(dim=1))
							 ]))

    # Connect pretrained model with modified classifer layer
    model.fc = fc 
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), 
                      lr=0.01,momentum=0.9,
                      nesterov=True,
                      weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # Send to GPU
    model.to(device)
    
    #Set checkpoint
    try:
        checkpoint = torch.load(CHECK_POINT_PATH)
        print("checkpoint loaded")
    except:
        checkpoint = None
        print("checkpoint not found")

    if checkpoint == None:
    

