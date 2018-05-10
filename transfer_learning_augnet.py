
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
from dataloader import *

# Basics transformations with no augmentations.
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


# Load the train and test data using seperate data loaders.
data_dir = 'SunData'
image_datasets = {}
image_datasets['train'] = TrainLoader(os.path.join(data_dir, 'train'), data_transforms['train'])
image_datasets['val'] = TestLoader(os.path.join(data_dir, 'val'), data_transforms['val'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print('dataset_sizes',dataset_sizes)
class_names = image_datasets['train'].classes

print("Data loaded")

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() 
        return x.view(N, -1)

# This function calls the train module.
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Keep a track of the best module and accuracy.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # dtype = torch.FloatTensor

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        

        # GO through the training and validation phase for each epoch.
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for input1, input2, labels in dataloaders[phase]:
                inp1 = input1.numpy()
                inp2 = input2.numpy()
                # Create a 6 channel input.
                inputs = np.stack((inp1, inp2), axis=1)
                if inputs.shape[0] == 1:
                    continue
                inputs = np.reshape(inputs, (2,6,224,224))
                inputs = torch.from_numpy(inputs)
                inputs = inputs.cuda()
                inputs = Variable(inputs)
                labels = labels.cuda()
                labels = Variable(labels)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                #Get the output and predictions from the model.
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # If we are in the training phase then back propogate.
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss 
                running_corrects += torch.sum(preds == labels)

            # Calculate the loss and accuracy per phase
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print("epoch_acc",epoch_acc)
            print("epoch_loss",epoch_loss)


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

    model.load_state_dict(best_model_wts)
    return model


# Define the classification model
smallnet_model = nn.Sequential(
    nn.Conv2d(6,3,kernel_size=3,stride=1),
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


# Define the augmentation model.
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

smallnet_model = smallnet_model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(smallnet_model.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print("Hyperparameters set, beginning training.")

smallnet_model = train_model(smallnet_model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=50)