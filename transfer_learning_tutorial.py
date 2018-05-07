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
from neural_style import  *
from dataloader import *

data_transforms = {
    'train': transforms.Compose([
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
image_datasets = {}
image_datasets['train'] = TrainLoader(os.path.join(data_dir, 'train'), data_transforms['train'])
image_datasets['val'] = TestLoader(os.path.join(data_dir, 'val'), data_transforms['val'])
print("Data loaded")

# # desired depth layers to compute style/content losses :
# content_layers_default = ['conv_4']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def train_model(model_smallnet, model_augnet, criterion, optimizer_smallnet, optimizer_augnet, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts_smallnet = copy.deepcopy(model_smallnet.state_dict())
    best_model_wts_augnet = copy.deepcopy(model_augnet.state_dict())
    best_acc = 0.0
    dtype = torch.FloatTensor

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Step 1: Train both the small net and augnet.

        scheduler.step()
        '''
        # ------------------
        #      PHASE 1
        # ------------------

        model_smallnet.train()

        for input1, input2, labels in dataloaders['train']:

            labels = labels.cuda()
            labels = Variable(labels)

            inputs = run_style_transfer(input1, input2, input1.clone())

            optimizer_smallnet.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model_smallnet(inputs)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer_smallnet.step()

         '''
        # ------------------
        #      PHASE 2
        # ------------------

        model_smallnet.train()
        model_augnet.train()

        running_loss = 0.0
        running_corrects = 0

        for input1, input2, labels in dataloaders['train']:
            inp1 = input1.numpy()
            inp2 = input2.numpy()
            inputs = np.vstack((inp1, inp2))
            inputs = torch.from_numpy(inputs)
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs = Variable(inputs)
            labels = Variable(labels)

            optimizer_smallnet.zero_grad()
            optimizer_augnet.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model_augnet(inputs)
                outputs = model_smallnet(outputs)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer_smallnet.step()
                optimizer_augnet.step()
                running_loss += loss
                running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print("PHASE 2:")
        print("loss", epoch_loss)
        print("accuracy", epoch_acc)
        

        # ------------------
        #      PHASE 3
        # ------------------
        '''

        model_smallnet.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['eval']:
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs = Variable(inputs)
            labels = Variable(labels)

            optimizer_smallnet.zero_grad()

            outputs = model_smallnet(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            running_loss += loss
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print("PHASE 3:")
        print("loss", epoch_loss)
        print("accuracy", epoch_acc)
    '''
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts_augnet = copy.deepcopy(model_augnet.state_dict())
        best_model_wts_smallnet = copy.deepcopy(model_smallnet.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model_smallnet.load_state_dict(best_model_wts_smallnet)
    model_augnet.load_state_dict(best_model_wts_augnet)
    return model_smallnet, model_augnet


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
    nn.Linear(53,5)
)

dtype = torch.FloatTensor

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

print("initialized final layer")

#smallnet_model = smallnet_model.cuda()
#augmented_model = augmented_model.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_smallnet = optim.SGD(smallnet_model.parameters(), lr=0.001, momentum=0.9)
optimizer_augnet = optim.SGD(smallnet_model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_smallnet, step_size=7, gamma=0.1)

print("Hyperparameters set, beginning training.")
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#

smallnet_model, augnet_model = train_model(smallnet_model, augmented_model,criterion, optimizer_smallnet, optimizer_augnet,exp_lr_scheduler,num_epochs=50)
