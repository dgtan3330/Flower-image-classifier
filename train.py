# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# References: Machine Learning A-Z: Hands-On Python & R in Data Science
# Additional reference from Mentors: https://realpython.com/command-line-interfaces-python-argparse/
# Additional reference from Mentors: https://docs.python.org/3/library/argparse.html#default
# Ardent Student of Python and AI, and inspiring developer: DG Tan

# load libraries
import argparse  
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict
import time
import numpy as np
import matplotlib.pyplot as plt
from earth import save_checkpoint, load_checkpoint




# Command Line Arguments as follows: 

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('data_dir', action='store', default='flowers') # change from '--data_dir'
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'], help='CNN architecture')
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001', help='learning rate')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512', help='hidden units')
    parser.add_argument('--epochs', dest='epochs', default='3', help='num of epochs')
    parser.add_argument('--gpu', action='store_true', default='gpu', help='use GPU for training')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth", help='save the trained model to a checkpoint')
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    print('training...')
    steps = 0
    print_every = 10
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders[0]): # 0 = train
            steps += 1 
            #if torch.cuda.is_available(): # testing this out, uncomment later
               # model.cuda()
            if gpu == 'gpu': # Optional to put in Main: device = torch.device('cuda' if torch.cuda.is_available() and in_args.gpu else 'cpu')
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda
            else:
                model.cpu() # use a CPU if user says anything other than "gpu"
            #inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda 
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valloss = 0
                accuracy=0

                for ii, (inputs2,labels2) in enumerate(dataloaders[1]): # 1 = validation 
                        optimizer.zero_grad()
                        
                        if gpu == 'gpu':
                            inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') # use cuda
                            model.to('cuda:0') # use cuda
                        else:
                            pass # just use inputs
                        #if torch.cuda.is_available(): 
                        with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            valloss = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                valloss = valloss / len(dataloaders[1])
                accuracy = accuracy /len(dataloaders[1])

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(valloss),
                      "Accuracy: {:.4f}".format(accuracy),
                     )

                running_loss = 0
            
def main():
    print("Test") # just to test, before training takes a long time
    args = parse_args()
# Dictionary holding location of training and validation data

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    image_size = 224 # Image size in pixels # Goes into Cell 5
    reduction = 256 # Image reduction to smaller edge # Goes into Cell 5
    norm_means = [0.485, 0.456, 0.406] # Normalized means of the images # Goes into Cell 5
    norm_std = [0.229, 0.224, 0.225] # Normalized standard deviations of the images # Goes rotation = 30 # Range of degrees for rotation # Goes into Cell 5
    rotation = 30 # Range of degrees for rotation # Goes into Cell 5
    batch_size = 64 # Number of images used in a single pass # Goes into Cell 5
    shuffle = True # Randomize image selection for a batch # Goes into Cell 5
    
    training_transforms = transforms.Compose([transforms.RandomRotation(rotation), transforms.RandomResizedCrop(image_size),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(reduction), transforms.CenterCrop(image_size), transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])]) 

    testing_transforms = transforms.Compose([transforms.Resize(reduction), transforms.CenterCrop(image_size), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])]) 

    image_datasets = [ImageFolder(train_dir, transform=training_transforms),
                      ImageFolder(val_dir, transform=validation_transforms),
                      ImageFolder(test_dir, transform=testing_transforms)]
    
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('drop', nn.Dropout(p=0.6)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss() # using criterion and optimizer, similar to points learnt in pytorch lectures (densenet)
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = image_datasets[0].class_to_idx
    gpu = args.gpu # get the gpu settings
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    path = args.save_dir # generate the new save location 
    save_checkpoint(path, model, optimizer, args, classifier)
    

if __name__ == "__main__":
    main()
    
print("Model is successfully trained") 
