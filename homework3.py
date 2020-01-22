import os
import sys
import json
import time
import logging
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
from torch.backends import cudnn

import torchvision
from torchvision import transforms
from torchvision.models import detection

from PIL import Image
from tqdm import tqdm

from weapons_dataset import WeaponS

# **Training function**
def training_procedure(net,images,labels,step,optimizer,criterion,alpha=None):
    global DEVICE,LOG_FREQUENCY
    # Bring data over the device of choice
    images = images.to(DEVICE)
    # print(type(labels).__name__)
    labels = labels.to(DEVICE)

    net.train() # Sets module in training mode
    # PyTorch, by default, accumulates gradients after each backward pass
    # We need to manually set the gradients to zero before starting a new iteration
    optimizer.zero_grad() # Zero-ing the gradients

    # Forward pass to the network
    outputs = net(images,alpha)

    # Compute loss based on output and ground truth
    loss = criterion(outputs, labels)

    # Log loss
    if step % LOG_FREQUENCY == 0:
        print('Step {}, Loss {}'.format(step, loss.item()))

    # Compute gradients for each layer and update weights
    loss.backward()  # backward pass: computes gradients
    return loss

def validation_procedure(net,dataLoader,len_dataset):
    net.train(False) # Set Network to evaluation mode
    running_corrects = 0.0
    for images, labels in dataLoader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward Pass
        outputs = net(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()
    # Calculate Accuracy
    return running_corrects / len_dataset



def main(train_dataset,test_dataset,validation_dataset):
    # To add a new cell, type ''
    # To add a new markdown cell, type '
    

    # **Set Arguments**
    time_mesure = time.time()

    # **Prepare Dataloaders**

    
    # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)


    # **Prepare Network**

    
    net = detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # **Prepare Training**

    
    # Define loss function
    criterion = nn.CrossEntropyLoss() # for classification, we use Cross Entropy

    # Choose parameters to optimize
    # To access a different set of parameters, you have to access submodules of AlexNet
    # (nn.Module objects, like AlexNet, implement the Composite Pattern)
    # e.g.: parameters of the fully connected layers: net.classifier.parameters()
    # e.g.: parameters of the convolutional layers: look at alexnet's source code ;) 
    parameters_to_optimize = net.parameters() # In this case we optimize over all the parameters of AlexNet
    # parameters_to_optimize = chain(net.features.parameters() , net.classifier[6].parameters())

    # Define optimizer
    # An optimizer updates the weights based on loss
    # We use SGD with momentum
    optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Define scheduler
    # A scheduler dynamically changes learning rate
    # The most common schedule is the step(-down), which multiplies learning rate by gamma every STEP_SIZE epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # By default, everything is loaded to cpu
    # saved_net = net
    net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
    cudnn.benchmark # Calling this optimizes runtime

    #initialSchedulerState = scheduler.state_dict()
    # for stepSize in [STEP_SIZE,STEP_SIZE-10,STEP_SIZE+10]:
    #   for lr in [i/1000 for i in range(6,80,5)]:
        # scheduler.load_state_dict(initialSchedulerState)
        # scheduler.base_lrs = [lr]
        # scheduler.stepSize = stepSize
        # net = saved_net.to(DEVICE)
    best_net = None
    best_score = None
    # Start iterating over the epochs
    current_step = 0
    for epoch in range(NUM_EPOCHS):
        print('Starting epoch {}/{}, LR = {}'.format(epoch+1, NUM_EPOCHS, scheduler.get_lr()))

        # Iterate over the dataset
        # for train, validation in zip(train_dataloader,validation_dataloader):
        #     train_images,train_labels = train
        #     validation_images , _ = validation
        for train_images,train_labels in train_dataloader:
            loss = training_procedure(net,train_images,train_labels,current_step,optimizer,criterion)
            # if loss < 0.06 and best_score > 0.3:
            #     training_procedure(net,train_images, torch.zeros(len(train_images), dtype=torch.long),current_step,optimizer,criterion, )
            #     training_procedure(net,validation_images,torch.tensor([1] * len(validation_images), dtype=torch.long),current_step,optimizer,criterion,ALPHA)
            optimizer.step() # update weights based on accumulated gradients
            current_step += 1

        accuracy = validation_procedure(net, validation_dataloader, float(len(validation_dataset)))
        if best_net is None or accuracy > best_score:
            best_net = net
            best_score = accuracy
            print('new best accuracy on validation {}'.format(accuracy))
        if epoch in range(5,NUM_EPOCHS+1,5):
            with open("results.json","a") as jsonFile:
                json.dump({"score":best_score,"learningRate":LR,"nbEpochs":epoch,"stepSize":STEP_SIZE,"gamma":GAMMA,"alpha":ALPHA},jsonFile)
                jsonFile.write("\n")
        # Step the scheduler
        scheduler.step()


    # **Test**

    
    net = best_net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
    net.train(False) # Set Network to evaluation mode

    running_corrects = 0
    for images, labels in tqdm(test_dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward Pass
        outputs = net(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / float(len(test_dataset))

    print('Test Accuracy: {}'.format(accuracy))
    with open("results.json","a") as jsonFile:
        json.dump({"score":accuracy,"learningRate":LR,"nbEpochs":epoch,"stepSize":STEP_SIZE,"alpha":ALPHA},jsonFile)
        jsonFile.write("\n")
    print('Time of execution {:.2f}s'.format(time.time()-time_mesure))


    
if __name__ == '__main__':
    global  DEVICE,DATA_DIR,BATCH_SIZE,ALPHA,LR,MOMENTUM,WEIGHT_DECAY,NUM_EPOCHS,STEP_SIZE,GAMMA,LOG_FREQUENCY
    DEVICE = 'cuda' # 'cuda' or 'cpu'
    BATCH_SIZE = 160     # Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
                        # the batch size, learning rate should change by the same factor to have comparable results
    LOG_FREQUENCY = 20
    MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
    WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default
    NUM_EPOCHS = 40      # Total number of training epochs (iterations over dataset)

    # **Define Data Preprocessing**

    
    # Define transforms for training phase
    norm_top = (0.485, 0.456, 0.406) # (0.5, 0.5, 0.5)
    norm_down = (0.229, 0.224, 0.225) # (0.5, 0.5, 0.5)
    train_transform = transforms.Compose([transforms.Resize(256),      # Resizes short size of the PIL image to 256
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomCrop(224),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                        transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                        transforms.Normalize(norm_top, norm_down) # Normalizes tensor with mean and standard deviation
    ])
    # Define transforms for the evaluation phase
    eval_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),#Crops a central square patch of the image
                                                                    #224 because torchvision's AlexNet needs a 224x224 input!
                                                                    #Remember this when applying different transformations, otherwise you get an error
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_top, norm_down)                                    
    ])

    # **Prepare Dataset**

    # Prepare Pytorch train/test Datasets
    DATA_DIR = 'Homework3-PACS/PACS'

    # Prepare Pytorch train/test Datasets
    train_dataset = torchvision.datasets.ImageFolder(DATA_DIR+"/photo", transform=train_transform)
    cartoon_dataset = torchvision.datasets.ImageFolder(DATA_DIR+"/cartoon", transform=eval_transform)
    sketch_dataset = torchvision.datasets.ImageFolder(DATA_DIR+"/sketch", transform=eval_transform)
    test_dataset = torchvision.datasets.ImageFolder(DATA_DIR+"/art_painting", transform=eval_transform)

    validation_dataset = list(chain(cartoon_dataset, sketch_dataset))
    validation_indexes = [idx for idx in range(len(validation_dataset)) if not idx % 3]
    validation_dataset = Subset(validation_dataset, validation_indexes)

    # Check dataset sizes
    print('Train Dataset: {}'.format(len(train_dataset)))
    print('validation Dataset: {}'.format(len(validation_dataset)))
    print('Test Dataset: {}'.format(len(test_dataset)))
    nbRun = 0
    for stepSize in range(10,30,10):
        STEP_SIZE = stepSize       # How many epochs before decreasing learning rate (if using a step-down policy)
        for g in range(5,7):
            GAMMA = g/10           # Multiplicative factor for learning rate step-down
            for a in range(10,30,5):
                ALPHA = a/100      # damping coefficient of the confusion network influence
                for lr in range(7,1,-2):
                    LR = lr/1000   # The initial Learning Rate
                    s = "=====START__RUN_{}_[STEP_SIZE:{},GAMMA:{},ALPHA:{},LR:{}]=====\n".format(nbRun,stepSize,g,a,lr)
                    with open("results.json","a") as f:
                        f.write(s)
                    print(s)
                    main(train_dataset,test_dataset,validation_dataset)
                    nbRun += 1