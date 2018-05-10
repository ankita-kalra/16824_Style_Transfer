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
from dataloader import TrainLoader, TestLoader
import scipy.misc as m

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
def train_model(model_smallnet, model_augnet, criterion, optimizer_smallnet, optimizer_augnet, scheduler, num_epochs=30):
    since = time.time()

    # Keep a track of the best module and accuracy.
    best_model_wts_smallnet = copy.deepcopy(model_smallnet.state_dict())
    best_model_wts_augnet = copy.deepcopy(model_augnet.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Step 1: Train both the small net and augnet.
        scheduler.step()

        # ------------------
        #      PHASE 1
        # ------------------
        print("PHASE 1")

        model_smallnet.train()

        for input1, input2, labels in dataloaders['train']:

            labels = labels.cuda()
            labels = Variable(labels)

            inputs = run_style_transfer(input1, input2, input1.clone())

            optimizer_smallnet.zero_grad()

            with torch.set_grad_enabled(True):
              inputs = inputs.cuda()

              outputs = model_smallnet(inputs)
              _, preds = torch.max(outputs, 1)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer_smallnet.step()


        # Step 2: 
        # Set the Augnet to Evaluation mode.
        # Calculate the style and content loss and backprop on the input image
        # Pass the output of the AugNet to the Classification Network.
        # ------------------
        #      PHASE 2
        # ------------------
        print("PHASE 2")

        model_smallnet.train()
        model_augnet.train()

        running_loss = 0.0
        running_corrects = 0

        for input1, input2, labels in dataloaders['train']:
            inp1 = input1.numpy()
            inp2 = input2.numpy()
            inputs = np.stack((inp1, inp2), axis=1)
            if inputs.shape[0] == 1:
                continue
            # print("0 bug source", inputs.shape)
            inputs = np.reshape(inputs, (2,6,224,224))
            inputs = torch.from_numpy(inputs)
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs = Variable(inputs)
            labels = Variable(labels)

            optimizer_smallnet.zero_grad()
            optimizer_augnet.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model_augnet(inputs)
                image = outputs.data.cpu().numpy()
                image = np.reshape(image, (2, 222, 222, 3))
                image1 = m.imresize(image[0, :, :, :], (224, 224), 'nearest')
                image2 = m.imresize(image[1, :, :, :], (224, 224), 'nearest')
                image = np.dstack((image1, image2))
                image = np.reshape(image, (2, 3, 224, 224))
                inputs = torch.from_numpy(image)
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.cuda()
                inputs = Variable(inputs) 
                outputs = model_smallnet(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_smallnet.step()
                optimizer_augnet.step()
                running_loss += loss
                running_corrects += torch.sum(preds == labels)
            

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        print("PHASE 2:")
        print("loss", epoch_loss)
        print("accuracy", epoch_acc)
	
        # Step 3: Evaluate the performance of the Classification Network on the test dataset.
        # ------------------
        #      PHASE 3
        # ------------------

        print("PHASE 3")

        model_smallnet.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['val']:
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
            break

        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects.double() / dataset_sizes['val']

        print("PHASE 3:")
        print("loss", epoch_loss)
        print("accuracy", epoch_acc.data.cpu().numpy())
    if epoch_acc.data.cpu().numpy()[0] > best_acc:
            best_acc = epoch_acc.data.cpu().numpy()[0]
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

# Define the classification model
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

# Define the augmentation model.
augmented_model = nn.Sequential(
    nn.Conv2d(6,16,kernel_size=3,stride=1),
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
augmented_model = augmented_model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer_smallnet = optim.SGD(smallnet_model.parameters(), lr=0.001, momentum=0.9)
optimizer_augnet = optim.SGD(smallnet_model.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_smallnet, step_size=7, gamma=0.1)

print("Hyperparameters set, beginning training.")

smallnet_model, augnet_model = train_model(smallnet_model, augmented_model, criterion, optimizer_smallnet, optimizer_augnet, exp_lr_scheduler, num_epochs=50)
