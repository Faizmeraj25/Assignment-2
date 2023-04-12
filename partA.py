import torch
import torchvision 
from torchvision import datasets
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
def load_dataset(data_augmentation , train_path, test_path, train_batch_size, val_batch_size, test_batch_size):
    transformer1 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4602, 0.4495, 0.3800], std=[0.2040, 0.1984, 0.1921])
    ])
    train_Dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transformer1)
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
        augmented_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transformer2)
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

class create_lenet(torch.nn.Module):
    def __init__(self, num_classes, Kernel_size, num_filters, activation_func, filter_factor, is_data_augment, dropout_factor):
        super(create_lenet, self).__init__()
        self.actfunc1 = None
        self.actfunc2 = None
        self.actfunc3 = None
        self.actfunc4 = None
        self.actfunc5 = None
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=Kernel_size)
        self.size = (256 - Kernel_size)
        prev_number_filters = num_filters
        number_filters = int(num_filters * filter_factor)
        if activation_func == 'ReLU':
            self.actfunc1 = torch.nn.ReLU()
        elif activation_func == 'GELU':
            self.actfunc1 = torch.nn.GELU()
        elif activation_func == 'SiLU':
            self.actfunc1 = torch.nn.SiLU()
        elif activation_func == 'Mish':
            self.actfunc1 = torch.nn.Mish()
        else:#By Default
            self.actfunc1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=Kernel_size, stride=2)
        self.size = self.size//2
        self.conv2 = torch.nn.Conv2d(in_channels=prev_number_filters, out_channels=number_filters, kernel_size=Kernel_size)
        # 50 x 68 x 68
        self.size = (self.size - Kernel_size) 
        prev_number_filters = number_filters
        number_filters = int(number_filters * filter_factor)
        
        if activation_func == 'ReLU':
            self.actfunc2 = torch.nn.ReLU()
        elif activation_func == 'GELU':
            self.actfunc2 = torch.nn.GELU()
        elif activation_func == 'SiLU':
            self.actfunc2 = torch.nn.SiLU()
        elif activation_func == 'Mish':
            self.actfunc2 = torch.nn.Mish()
        else:#By Default
            self.actfunc2 = torch.nn.ReLU()
            
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=Kernel_size, stride=(2,2))
        self.size = self.size//2
        
        self.conv3 = torch.nn.Conv2d(in_channels=prev_number_filters, out_channels=number_filters, kernel_size=Kernel_size)
        # 50 x 32 x 32
        self.size = (self.size - Kernel_size) 
        
        prev_number_filters = number_filters
        number_filters = int(number_filters * filter_factor)
        
        if activation_func == 'ReLU':
            self.actfunc3 = torch.nn.ReLU()
        elif activation_func == 'GELU':
            self.actfunc3 = torch.nn.GELU()
        elif activation_func == 'SiLU':
            self.actfunc3 = torch.nn.SiLU()
        elif activation_func == 'Mish':
            self.actfunc3 = torch.nn.Mish()
        else:#By Default
            self.actfunc3 = torch.nn.ReLU()
            
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=Kernel_size, stride=2)
        self.size = self.size//2
        
        self.conv4 = torch.nn.Conv2d(in_channels=prev_number_filters, out_channels=number_filters, kernel_size=Kernel_size)
        self.size = (self.size - Kernel_size) 
        
        prev_number_filters = number_filters
        number_filters = int(number_filters * filter_factor)
        
        if activation_func == 'ReLU':
            self.actfunc4 = torch.nn.ReLU()
        elif activation_func == 'GELU':
            self.actfunc4 = torch.nn.GELU()
        elif activation_func == 'SiLU':
            self.actfunc4 = torch.nn.SiLU()
        elif activation_func == 'Mish':
            self.actfunc4 = torch.nn.Mish()
        else:#By Default
            self.actfunc4 = torch.nn.ReLU()
            
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=Kernel_size, stride=2)
        self.size = self.size//2
        
        self.conv5 = torch.nn.Conv2d(in_channels=prev_number_filters, out_channels=number_filters, kernel_size=Kernel_size)
        self.size = (self.size - Kernel_size) 
        prev_number_filters = number_filters
        if activation_func == 'ReLU':
            self.actfunc5 = torch.nn.ReLU()
        elif activation_func == 'GELU':
            self.actfunc5 = torch.nn.GELU()
        elif activation_func == 'SiLU':
            self.actfunc5 = torch.nn.SiLU()
        elif activation_func == 'Mish':
            self.actfunc5 = torch.nn.Mish()
        else:#By Default
            self.actfunc5 = torch.nn.ReLU()
            
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=Kernel_size, stride=2)
        self.size = self.size//2
        #Need to calculate the in_features.
        
        self.size = self.size * self.size * prev_number_filters
        self.fc1 = torch.nn.Linear(in_features=self.size, out_features=self.size)
        self.dropp1 = torch.nn.Dropout(dropout_factor)
        self.fc2 = torch.nn.Linear(in_features=self.size, out_features=num_classes)
        self.dropp2 = torch.nn.Dropout(dropout_factor)
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.actfunc1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)

        x = self.actfunc2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.actfunc3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.actfunc4(x)
        x = self.maxpool4(x)

        
        x = self.conv5(x)
        x = self.actfunc5(x)
        x = self.maxpool5(x)
        x = x.view(-1,x.shape[1]*x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = self.dropp1(x)
        x = self.fc2(x)
        x = self.dropp2(x)
        output = self.logSoftmax(x)
        return x

def get_train_test_count(train_pat, test_pat):
    train_count = len(glob.glob(train_path+'/**/*.jpg'))
    test_count = len(glob.glob(test_path+'/**/*.jpg'))
    print("Training dataset count : ", train_count)
    print("Validation dataset count", test_count)
    return train_count, test_count


def train(cnn, learning_rate, epochs, train_Loader, val_Loader, train_count, test_count, is_wandb_log): 
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=cnn.parameters(), lr=learning_rate, weight_decay = 1e-4)

    for epoch in range(epochs):
        train_accuracy = 0
        train_loss = 0
        cnn.train()
        for i, (images, labels) in enumerate(train_Loader):

            images, labels = images.to(device), labels.to(device)
            # doing zero gradient.
            optimizer.zero_grad()

            #forward Propagation
            y_pred = cnn(images)

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
            cnn.eval()
            for i, (images, labels) in enumerate(val_Loader):
                images, labels = images.to(device), labels.to(device)

                y_pred = cnn(images)

                loss = loss_function(y_pred, labels)
                test_loss += loss.item()

                _, predicted = torch.max(y_pred.data, 1)

                test_accuracy += int(torch.sum(predicted == labels.data))

            test_accuracy /= test_count
            test_loss /= test_count

            print(f"Epochs : {epoch+1} Validation Accuracy : {test_accuracy} Validation Loss {test_loss}")
            if(is_wandb_log):
                wandb.log({"train_accuracy": train_accuracy, "train_loss" : train_loss, "val_accuracy": test_accuracy, "val_error": test_loss}) 
            



def main(num_classes, kernel_size,  train_path, test_path, train_batch_size, val_batch_size, test_batch_size, num_filters, activation_func, filter_factor, is_data_augment, learning_rate, epochs, is_wandb_log, dropout_factor):
    print("train_path", train_path)
    print("test_path", test_path)
    train_Loader, val_Loader, test_Loader = load_dataset(is_data_augment, train_path, test_path, train_batch_size, val_batch_size, test_batch_size)
    cnn = create_lenet(num_classes, kernel_size, num_filters, activation_func, filter_factor, is_data_augment, dropout_factor)
    cnn = cnn.to(device)
    train_count, test_count = get_train_test_count(train_path, test_path)
    print("Training the Model...")
    train(cnn, learning_rate, epochs, train_Loader, val_Loader, train_count, test_count, is_wandb_log)
    print("Training Finished !!")

def plot_test_images():
    import matplotlib.pyplot as plt
    test_path = '/kaggle/input/inaturalist12k/Data/inaturalist_12K/val/'
    transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(), 
        ])

    test_Dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transformer)
    train_Loader = DataLoader(
    test_Dataset,
    batch_size=128, 
    shuffle=True)

    # Define the size of the grid
    nrow, ncol = 3, 10

    # Get some random images and their labels from the DataLoader
    # dataiter = iter(train_Loader)
    images, labels = next(iter(train_Loader))

    # Create the grid of images and labels
    grid = vutils.make_grid(images[:nrow*ncol], nrow=nrow, normalize=True, scale_each=True)
    grid = grid.permute(1, 2, 0).numpy()

    # Create a subplot to display the grid
    fig, ax = plt.subplots(figsize=(8, 30))
    ax.imshow(grid)

    # Add labels to the subplot
    nrow, ncol = ncol, nrow
    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            x = j*grid.shape[1]//ncol
            y = (i+1)*grid.shape[0]//nrow - 20
            ax.text(x, y, classes[labels[idx]], ha='left', va='top', color='red')

    # Hide axis and show the plot
    ax.axis('off')
    plt.show()

def run_sweep():
    
    default_params =dict(
        num_filters = 32, 
        activation_func = 'ReLU',
        filter_factor = 2, 
        learning_rate = 0.001,
        epochs = 3,   
        dropout_factor = 0,
    )

    run=wandb.init(config=default_params,project=wandb_project,entity=wandb_entity,reinit='true')

    config = wandb.config

    num_filters = config.num_filters
    activation_func = config.activation_func
    filter_factor = config.filter_factor
    learning_rate = config.learning_rate
    epochs = config.epochs
    dropout_factor = config.dropout_factor

    run.name = 'ac_' + activation_func + '_nf_' + str(num_filters) + '_ff_'+ str(filter_factor)+'_df_'+str(dropout_factor)
    main(num_classes, kernel_size,  train_path, test_path, train_batch_size, val_batch_size, test_batch_size, num_filters, activation_func, filter_factor, is_data_augment, learning_rate, epochs, is_wandb_log, dropout_factor)



import wandb
train_path = 'nature_12k/inaturalist_12K/train/'
test_path = 'nature_12k/inaturalist_12K/val/'
dir_path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(dir_path, train_path)
test_path = os.path.join(dir_path, test_path)
train_batch_size = 64
test_batch_size = 16
val_batch_size = 16
num_classes = 10
kernel_size = 3
is_wandb_log = True
is_data_augment=True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#Set up a sweep config
sweep_config = {
    'metric': {
        'goal': 'maximize', 
        'name': 'val_accuracy'
        },
    "method": "bayes",
    "project": "Deep_Learning_Assignment2",
    "parameters": {
        "num_filters": {
            "values": [32, 64]
        },
        "filter_factor": {
            "values": [0.5, 1, 2]
        },
        "epochs": {
            "values": [5, 10]
        },
        "activation_func":
        {
            "values" : ["ReLU", "GELU", "SiLU", "Mish"]
        },
        "dropout_factor":
        {
            "values": [0 , 0.3, 0.4]
        },
    }
}

# creating the sweep
sweep_id = wandb.sweep(sweep_config, project=wandb_project)
print('sweep_id', sweep_id)
wandb.agent(sweep_id, function=run_sweep)



####################   RUN WITHOUT SWEEP ###############################
# train_path = 'inaturalist_12K/train/'
# test_path = 'inaturalist_12K/val/'
# dir_path = os.path.dirname(os.path.realpath(__file__))
# train_path = os.path.join(dir_path, train_path)
# test_path = os.path.join(dir_path, test_path)
# print('train_path', train_path)
# print(dir_path)
# train_batch_size = 64
# test_batch_size = 16
# val_batch_size = 16
# num_classes = 10
# kernel_size = 3
# is_wandb_log = False
# is_data_augment=True
# num_filters = 128
# activation_func = 'Mish'
# filter_factor = 2
# learning_rate = 0.001
# epochs = 3
# dropout_factor = 0.3

# main(num_classes, kernel_size,  train_path, test_path, train_batch_size, val_batch_size, test_batch_size, num_filters, activation_func, filter_factor, is_data_augment, learning_rate, epochs, is_wandb_log, dropout_factor)
