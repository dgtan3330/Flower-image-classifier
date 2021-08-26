# Basic usage: python predict.py data_directory
# Uses a trained network to predict the class for an input image
# References: Machine Learning A-Z: Hands-On Python & R in Data Science
# Additional reference from Mentors: https://realpython.com/command-line-interfaces-python-argparse/
# Additional reference from Mentors: https://docs.python.org/3/library/argparse.html#default
# Ardent Student of Python and AI, and inspiring developer: DG Tan

import argparse       
import tensorflow as tf
import torch
from torchvision import transforms, datasets
import copy
import os
import json




def save_checkpoint(path, model, optimizer, args, classifier):
    
    checkpoint = {'arch': args.arch, 
                  'model': model,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier' : classifier,
                  'epochs': args.epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, path) # User-defined path, otherwise set to checkpoint.pth
    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names