# Basic python imports for logging and sequence generation
import itertools
import random
import logging
import pickle
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np

# Imports for Pytorch for the things we need
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms, datasets
from cifar_pytorch import Net, CIFAR3

#globals
PATH = './cifarBestRun.pth'

def main():  
  test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  test_data = CIFAR3("test", transform=test_transform)
  batch_size = 256
  testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
  testNet = Net() #Q3
  testNet.to('cpu')
  testNet.load_state_dict(torch.load(PATH))  
  
  correct = 0
  total = 0
  with torch.no_grad(): #not training, no gradients needed
    for data in testloader:
      images, labels = data
      outputs = testNet(images)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  
  print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

if __name__=="__main__":
  main()
