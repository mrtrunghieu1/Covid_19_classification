'''Main function to demo covid classification'''

# Necessary packages
import os 
import re
import PIL
import torch
import timm
import argparse
from torchvision import transforms, models, datasets
from randaugment import RandAugment, ImageNetPolicy, Cutout
import torch.nn as nn
# My packages

from data_helper import data_dir, CHECK_POINT_PATH
from training import create_model, train_model
from utils import load_model

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
    epochs = args.epochs

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
    fc = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(2048, 1000, bias=True)),
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
        CHECK_POINT_PATH = CHECK_POINT_PATH
        
    model, best_val_loss, best_val_acc = train_model(model, criterion, optimizer, scheduler, num_epochs=epochs, checkpoint=torch.load(CHECK_POINT_PATH))
    torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_acc,
            'scheduler_state_dict': scheduler.state_dict(),
            }, CHECK_POINT_PATH)
    
    # Load checkpoint path 
    try:
        checkpoint = torch.load(CHECK_POINT_PATH)
        print("checkpoint loaded")
    except:
        checkpoint = None
        print("checkpoint not found")

    load_model(model= model, checkpoint= checkpoint, path=CHECK_POINT_PATH)


    # Testing
    since = time.time()
    model.eval()
    y_test = []
    y_pred = []
    for images, labels in data_loader['test']:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = model(images)
        _, predictions = outputs.max(1)
        
        y_test.append(labels.data.cpu().numpy())
        y_pred.append(predictions.data.cpu().numpy())
        
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)
    pd.DataFrame({'true_label':y_test,'predicted_label':y_pred}).to_csv('Modified_EfficienNet_B0_Covid-19_Test.csv', index=False)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    sns.heatmap(confusion_matrix(y_test, y_pred))
    accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)
    print(report)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='batch_size',
        default=16,
        type=int
    )
    parser.add_argument(
        '--epochs',
        help='epochs',
        default=30,
        type=int
    )
    parser.add_argument(
        '--flag_model',
        help='flag_model',
        default=0,
        type=int
    )

    args = parser.parse_args()
    # Call main function
    main(args)
