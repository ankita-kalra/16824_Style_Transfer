from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'SunData'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print("Data loaded")


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs
                labels = labels
                inputs = Variable(inputs)
                labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                # with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss[0] * inputs.size(0)
                # print(type(preds))
                # print(type(labels.data))
                # print(preds.cpu() == labels.data.cpu())
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Load a pretrained model and reset final fully connected layer.
#

# model_ft = models.resnet18(pretrained=True)
#-------------> SMALL NET
#TODO  - Add Smallnet
# 3x3 Convolution

smallnet_model = nn.Sequential(
	nn.Conv2d(3,16,kernel_size=3,stride=1),
	nn.ReLU(inplace=True),
	nn.MaxPool2d(2,stride=2),
	nn.Conv2d(16,32,kernel_size=3,stride=1),
	nn.ReLU(inplace=True),
	nn.Conv2d(32,32,kernel_size=3,stride=1),
	nn.ReLU(inplace=True),
	nn.BatchNorm2d(32),
	nn.MaxPool2d(2,stride=2),
	Flatten(),
	nn.Linear(89888,53),
	nn.Dropout(0.5),
	nn.Linear(53,2)
)


dtype = torch.FloatTensor
x = torch.randn(4, 3, 224, 224).type(dtype)
x_var = Variable(x).type(dtype)
outputs = smallnet_model(x_var)
print(outputs.shape)

'''
print("Pretrained model collected.")
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 5)
'''
'''
model_ft = models.alexnet(pretrained=True)
#TODO - Augmentation Network
print("Pretrained model collected.")
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 5)
'''
'''
augmented_model = nn.Sequential(
	nn.Conv2d(2,16,kernel_size=3,stride=1),
	nn.ReLU(inplace=True),
	nn.Conv2d(16,16,kernel_size=3,stride=1),
	nn.ReLU(inplace=True),
	nn.Conv2d(16,16,kernel_size=3,stride=1),
	nn.ReLU(inplace=True),
	nn.Conv2d(16,16,kernel_size=3,stride=1),
	nn.Conv2d(16,3,kernel_size=3,stride=1)
	)
'''


print("initialized final layer")

#smallnet_model = smallnet_model.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(smallnet_model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print("Hyperparameters set, beginning training.")
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#

#smallnet_model = train_model(smallnet_model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=50)
