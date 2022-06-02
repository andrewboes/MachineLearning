# Basic python imports for logging and sequence generation
import itertools
import random
import logging
import numpy as np
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# Imports for Pytorch for the things we need
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from random import randrange

# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


# Set random seed for python and torch to enable reproducibility (at least on the same hardware)
random.seed(42)
torch.manual_seed(42)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #hack
x = 1152
y = 720
maxSlope = .25
maxLineSamples = 8
minLineSamples = 4
maxSamplesPerTimeStep = 4

def main():
  train = Lines(split='train')
  print(train.data)
  

def getLines(numLines):
  noise = np.random.normal(0,1,100)
  lines = []
  for i in range(0, numLines):
    b = randrange(200, y)
    m = ((randrange(maxSlope * 100 * 2))/100)-maxSlope
    numSamples = randrange(minLineSamples,maxLineSamples)
    x_0 = x/numSamples
    line = []
    for j in range(1, numSamples):
      x_t = x_0*(j+1)+noise[j]
      line.append([x_t, (x_t*m + b)-noise[j]])
    lines.append(line)
  return lines

class Lines(Dataset):
  
  def __init__(self,split="train", max_length=4):
    lines = getLines(max_length)
    print(lines)
    currentLineIndex = [-1,-1]
    self.data = {}
    self.data[0] = []
    for i in range(1, max_length*maxLineSamples):
      self.data[i]=[]
      currentLineIndex[1] = currentLineIndex[1] + 1
      currentLineIndex[0] = -1
      for j in range(0, 1):
        currentLineIndex[0] += 1
        if len(lines)>currentLineIndex[0] and len(lines[currentLineIndex[0]]) > currentLineIndex[1]: #is there entry
          pointToAdd = lines[currentLineIndex[0]][currentLineIndex[1]]
          if len(self.data[i]) == 0:
            self.data[i] = [pointToAdd]  
          else:
            self.data[i].append(pointToAdd)
      
    

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      
      x = self.data[idx]
      y = x.sum() % 2
      return x,y 
main()
