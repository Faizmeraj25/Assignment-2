#Importing the required libraries
import torch
import torchvision 
from torchvision.datasets import FashionMNIST
from torchvision import datasets, models
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

import argparse
# Implemented Arg parse to take input of the hyperparameters from the command.
parser = argparse.ArgumentParser(description="Stores all the hyperpamaters for the model.")
parser.add_argument("-wp", "--wandb_project",type=str, default="Deep_Learning_Assignment2", help="Enter the Name of your Wandb Project")
parser.add_argument("-we", "--wandb_entity",type=str, default="cs22m081", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("-e", "--epochs",default="1", type=int, help="Number of epochs to train neural network.")
parser.add_argument("-nf", "--num_filters",default="3", type=int, help="Number of filters in the convolutianal neural network.")
parser.add_argument("-lr", "--learning_rate",default="0.001", type=float, help="Learning rate used to optimize model parameters")
parser.add_argument("-af", "--activ_func",default="ReLU", type=str, choices=["ReLU", "GELU", "Mish", "SiLU"])
parser.add_argument("-df", "--dropout_factor",default="0.3", type=float, help="Dropout factor in the cnn")
parser.add_argument("-ff", "--filter_factor",default="1", type=float, choices=[1, 0.5, 2])

args = parser.parse_args()

wandb_project = args.wandb_project
wandb_entity = args.wandb_entity
epochs = args.epochs
num_filters = args.num_filters
learning_rate = args.learning_rate
filter_factor = args.filter_factor
dropout_factor = args.dropout_factor

print(wandb_project, wandb_entity, epochs, num_filters, learning_rate, filter_factor, dropout_factor)

## function to load dataset and create a dataloader for train as well as test.
def load_dataset(data_augmentation , train_path, test_path, train_batch_size, val_batch_size, test_batch_size):
    transformer1 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4602, 0.4495, 0.3800], std=[0.2040, 0.1984, 0.1921])
    ])
    train_Dataset = torchvision.datasets.ImageFolder(train_path, transform=transformer1)
    train_datasize = int(0.8 * len(train_Dataset))
    train_Dataset, val_Dataset = random_split(train_Dataset, [train_datasize, len(train_Dataset) - train_datasize])
    if data_augmentation == True: 
        transformer2 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5), 
        transforms.RandomVerticalFlip(0.02),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4602, 0.4495, 0.3800], std=[0.2040, 0.1984, 0.1921])
        ])
        augmented_dataset = torchvision.datasets.ImageFolder(train_path, transform=transformer2)
        augmented_dataset_size = int(0.2 * len(augmented_dataset))
        augmented_dataset, _  =  random_split(augmented_dataset, [augmented_dataset_size, len(augmented_dataset) - augmented_dataset_size])
        train_Dataset = ConcatDataset([train_Dataset, augmented_dataset])
    train_Loader = DataLoader(
        train_Dataset, 
        batch_size = train_batch_size,
        shuffle=True)
    test_Loader = DataLoader(
        test_path,
        batch_size=test_batch_size, 
        shuffle=True)
    val_Loader = DataLoader(
        val_Dataset, 
        batch_size=val_batch_size, 
        shuffle=True)
    return train_Loader, val_Loader, test_Loader

# Function to train the model 
def train(model, learning_rate, epochs, train_Loader, val_Loader, train_count, test_count, is_wandb_log): 
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay = 1e-4)

    for epoch in range(epochs):
        train_accuracy = 0
        train_loss = 0
        model.train()
        for i, (images, labels) in enumerate(train_Loader):

            images, labels = images.to(device), labels.to(device)
            # doing zero gradient.
            optimizer.zero_grad()

            #forward Propagation
            y_pred = model(images)

            # Calculating Loss.
            loss = loss_function(y_pred, labels)

            # Backward Propagation
            loss.backward()

            # update rule
            optimizer.step()

            train_loss += loss.item()

            _, prediction = torch.max(y_pred.data, 1)
            train_accuracy += int(torch.sum(prediction == labels.data))
    
        train_accuracy /= train_count
        train_loss /= train_count
        print(f"Epochs : {epoch+1} Train Accuracy : {train_accuracy} Train Loss {train_loss}")
    
        test_accuracy = 0
        test_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (images, labels) in enumerate(val_Loader):
                images, labels = images.to(device), labels.to(device)

                y_pred = model(images)

                loss = loss_function(y_pred, labels)
                test_loss += loss.item()

                _, predicted = torch.max(y_pred.data, 1)

                test_accuracy += int(torch.sum(predicted == labels.data))

            test_accuracy /= test_count
            test_loss /= test_count

            print(f"Epochs : {epoch+1} Validation Accuracy : {test_accuracy} Validation Loss {test_loss}")
            if(is_wandb_log):
                wandb.log({"train_accuracy": train_accuracy, "train_loss" : train_loss, "val_accuracy": test_accuracy, "val_error": test_loss}) 
            

# function which returns the number of dataset images in train and test directory.
def get_train_test_count(train_path, test_path):
    train_count = len(glob.glob(train_path+'/**/*.jpg'))
    test_count = len(glob.glob(test_path+'/**/*.jpg'))
    print("Training dataset count : ", train_count)
    print("Validation dataset count", test_count)
    return train_count, test_count

train_path = 'nature_12k/inaturalist_12K/train/'
test_path = 'nature_12k/inaturalist_12K/val/'
dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir_path", dir_path)

train_path = os.path.join(dir_path, train_path)
test_path = os.path.join(dir_path, test_path)
print("train_path", train_path)

train_batch_size = 64
test_batch_size = 16
val_batch_size = 16
is_Data_Augmentation = True
train_Loader, val_Loader, test_Loader = load_dataset(is_Data_Augmentation, train_path, test_path, train_batch_size, val_batch_size, test_batch_size)
print(len(train_Loader), len(val_Loader), len(test_Loader))   

# setting device to 'cuda' if GPU is available else it is set to 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the Alexnet model with the pretrained weights. 
vggnet = torchvision.models.vgg16(weights=True)

# Freezing all the layeres of the model.
for param in alexnete.parameters():
    param.requires_grad = False

# Modifying the 6th layer of the model to address the change in the size of the output layer.
# Changing the output layer size to the number of classes available.
vggnet.classifier[6] = torch.nn.Linear(in_features = 4096, out_features = 10)
# Adding a softmax layer. 
vggnet.classifier.add_module("7", torch.nn.LogSoftmax(dim=1))
vggnet.to(device)
train_count, test_count = get_train_test_count(train_path, test_path)
learning_rate = 0.001
epochs = 20
is_wandb_log = False

# Training on the train dataset. 
train(vggnet, learning_rate, epochs, train_Loader, val_Loader, train_count, test_count, is_wandb_log)