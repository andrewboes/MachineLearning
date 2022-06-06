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

#matrix where the columns are the time steps and the rows are the number of boats

def main():
# =============================================================================
#   t = torch.tensor([[.1,0,0,1],[.2,0,0,1],[.3,0,0,1]]) #[X][y]
#   print(t)
#   print(t[:,:-1]) # x
#   print(t[:,-1:]) # y
# =============================================================================
  
  
  logging.info("Building model")
  input_size = 6 #number of features
  hidden_size = 100 #number of features in hidden state
  num_layers = 100 #number of stacked lstm layers
  maximum_training_sequence_length = 5
  
  train = Lines(split='train')  
# =============================================================================
#   print(train.data)
#   print(train.data[0].size())
#   print(train.data[1].size())
# =============================================================================
  train_loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=pad_collate)
      
  model = ParityLSTM(input_size, hidden_size, num_layers,110)
  model.to(torch.device("cpu"))
  
  logging.info("Model parameter info")
  for name, value in model.named_parameters(recurse=True):
      if value.requires_grad:
          print(name, value.size())#parameter meta data
          #print(value.data)#parameter values
  
  logging.info("Training model")
  train_model(model, train_loader)

class ParityLSTM(torch.nn.Module) :

    # __init__ builds the internal components of the model (presumably an LSTM and linear layer for classification)
    # The LSTM should have hidden dimension equal to hidden_dim
    
    def __init__(self, input_size, hidden_size, num_layers, outputSize) :
      super().__init__()
      self.hidden_size = hidden_size
      self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) #lstm
      self.fc_1 =  nn.Linear(hidden_size, outputSize) #fully connected 1
      
      #initial hidden and cell state values
      self.hiddenState = torch.nn.Parameter(torch.zeros(size=(self.lstm.num_layers, hidden_size)))
      self.cellState = torch.nn.Parameter(torch.zeros(size=(self.lstm.num_layers, hidden_size)))
                
    
    # forward runs the model on an B x max_length x 1 tensor and outputs a B x 2 tensor representing a score for 
    # even/odd parity for each element of the batch
    # 
    # Inputs:
    #   x -- a batch_size x max_length x 1 binary tensor. This has been padded with zeros to the max length of 
    #        any sequence in the batch.
    #   s -- a batch_size x 1 list of sequence lengths. This is useful for ensuring you get the hidden state at 
    #        the end of a sequence, not at the end of the padding
    #
    # Output:
    #   out -- a batch_size x 2 tensor of scores for even/odd parity    

    def forward(self, x, s):
      assert len(s) == x.size()[0] #test to ensure we're using the right shapes
      
      hiddenState = self.hiddenState.unsqueeze(1)#.expand(-1, len(x), -1) # update hidden dim
      cellState = self.cellState.unsqueeze(1)#.expand(-1, len(x), -1) # update cell dim
      #x = x.unsqueeze(-1) 
      xPacked = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False) #pack padded values of x 

      output, (hidden, cell) = self.lstm(xPacked, (hiddenState, cellState)) 
      return F.softmax(self.fc_1(hidden[-1]), dim=-1)

    def __str__(self):
        return "LSTM-"+str(self.hidden_size)

######################################################################
   
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
      x_t = x_0*(j+1)+noise[j]*2
      line.append(([float(x_t), float((x_t*m + b)-noise[j]*3)]))
    lines.append(line)
  return lines
  
class Lines(Dataset):
  
  def __init__(self,split="train", max_length=50):
    self.data = []
    for k in range(max_length):
      sample=[]  
      runningCount = 0
      numBoats = randrange(5,10)
      for i in range(numBoats):
        oneLine = getLines(1)[0]
        runningCount = runningCount + 1
        newArray = []
        for j in range(len(oneLine)):
          x = oneLine[j][0]
          y = oneLine[j][1]
          newArray.append(np.array([x,y,0,0,0,0,runningCount]))
        temp =np.vstack(newArray)
        if len(sample) == 0:
          sample = temp
        else:
          sample = np.vstack([sample, temp])
      self.data.append(torch.tensor(sample))
      
  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
    
    t = self.data[idx]
    x = t[:,:-1]
# =============================================================================
#     x = np.matrix.flatten(np.asarray(x))
#     x = torch.tensor(x).to(torch.int64)
# =============================================================================
    y = t[:,-1:][-1].squeeze()
    return x,y 

# Function to enable batch loader to concatenate binary strings of different lengths and pad them
def pad_collate(batch):
      (xx, yy) = zip(*batch)
      x_lens = [len(x) for x in xx]

      xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
      yy = torch.tensor(yy).long()

      return xx_pad, yy, x_lens

# Basic training loop for cross entropy loss
def train_model(model, train_loader, epochs=1000, lr=0.003):
    # Define a cross entropy loss function
    crit = torch.nn.CrossEntropyLoss()

    # Collect all the learnable parameters in our model and pass them to an optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # Adam is a version of SGD with dynamic learning rates 
    # (tends to speed convergence but often worse than a well tuned SGD schedule)
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.00001)

    # Main training loop over the number of epochs
    for i in range(epochs):
        
        # Set model to train mode so things like dropout behave correctly
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0

        # for each batch in the dataset
        for j, (x, y, l) in enumerate(train_loader):
            # predict the parity from our model
            y_pred = model(x.float(), l)
            #print(y_pred.size()) #torch.Size([1600, 1])
            
            # compute the loss with respect to the true labels
            loss = crit(y_pred, y)
            
            # zero out the gradients, perform the backward pass, and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute loss and accuracy to report epoch level statitics
            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        if i % 10 == 0:
            logging.info("epoch %d train loss %.3f, train acc %.3f" % (i, sum_loss/total, correct/total))#, val_loss, val_acc))

main()
