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
import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #hack

# Imports for Pytorch for the things we need
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms, datasets

# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt

debugMessages = False


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
    self.fc1 = nn.Linear(4096, 512)
    self.fc2 = nn.Linear(512, 3)  

  def excuteAndPrint(self, function, x):
    x = function(x)
    if debugMessages:
      print("{:}, after {:}".format(x.size(), function))
    return x

  def forward(self, x):
    x = self.excuteAndPrint(self.conv1, x)
    x = self.excuteAndPrint(F.relu, x) #relu1
    x = self.excuteAndPrint(self.conv2, x)
    x = self.excuteAndPrint(self.pool, x)
    x = self.excuteAndPrint(F.relu, x) #relu2
    x = self.excuteAndPrint(self.conv3, x)
    x = self.excuteAndPrint(F.relu, x) #relu3
    x = self.excuteAndPrint(self.conv4, x)
    x = self.excuteAndPrint(F.relu, x) #relu4
    x = self.excuteAndPrint(self.pool, x)
    x = x.view(-1, self.num_flat_features(x))
    if debugMessages:
      print(x.size())
    x = self.excuteAndPrint(self.fc1, x)
    x = self.excuteAndPrint(F.relu, x) #relu5
    x = self.excuteAndPrint(self.fc2, x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features



class CIFAR3(Dataset):

    def __init__(self,split="train",transform=None):
      if split=="train":
        with open("cifar10_hst_train", 'rb') as fo:
          self.data = pickle.load(fo) 
      elif split=="val":
        with open("cifar10_hst_val", 'rb') as fo:
          self.data = pickle.load(fo)
      else:
        with open("cifar10_hst_test", 'rb') as fo:
          self.data = pickle.load(fo)
      
      self.transform = transform

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):
        
        x = self.data['images'][idx,:]
        r = x[:1024].reshape(32,32)
        g = x[1024:2048].reshape(32,32)
        b = x[2048:].reshape(32,32)
        
        x = Tensor(np.stack([r,g,b]))

        if self.transform is not None:
          x = self.transform(x)
        
        y = self.data['labels'][idx,0]
        return x,y 






#########################################################
# Training and Evaluation
#########################################################
def main():
  transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
  train_data = CIFAR3("train", transform=transform)
  val_data = CIFAR3("val", transform=transform)
  test_data = CIFAR3("test", transform=transform)
  
  batch_size = 256
  trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
  valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
  testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
  
  
  # Build model
  model = Net()
  
  
  # Main training loop
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
  criterion = torch.nn.CrossEntropyLoss()
  
  device = torch.device('cpu')
  
  model.to(device)
  
  loss_log = []
  acc_log = []
  val_acc_log = []
  val_loss_log = []
  
  for i in range(50):
  
    # Run an epoch of training
    train_running_loss = 0
    train_running_acc = 0
    model.train()
    for j,input in enumerate(trainloader,0):
     
      x = input[0].to(device)
      y = input[1].type(torch.LongTensor).to(device)
      out = model(x)
      loss = criterion(out,y)
  
      model.zero_grad()
      loss.backward()
  
      optimizer.step()
  
      _, predicted = torch.max(out.data, 1)
      correct = (predicted == y).sum()
  
      train_running_loss += loss.item()
      train_running_acc += correct.item()
      loss_log.append(loss.item())
      acc_log.append(correct.item()/len(y))
  
    train_running_loss /= j
    train_running_acc /= len(train_data)
  
    # Evaluate on validation
    val_acc = 0
    val_loss = 0
    model.eval()
    for j,input in enumerate(valloader,0):
  
      x = input[0].to(device)
      y = input[1].type(torch.LongTensor).to(device)
  
      
      out = model(x)
  
      loss = criterion(out,y)
      _, predicted = torch.max(out.data, 1)
      correct = (predicted == y).sum()
  
      val_acc += correct.item()
      val_loss += loss.item()
  
    val_acc /= len(val_data)
    val_loss /= j
  
    val_acc_log.append(val_acc)
    val_loss_log.append(val_loss)
  
    logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(i,train_running_loss, train_running_acc*100,val_acc*100))
  
  
  # Plot training and validation curves
  fig, ax1 = plt.subplots(figsize=(16,9))
  color = 'tab:red'
  ax1.plot(range(len(loss_log)), loss_log, c=color, alpha=0.25, label="Train Loss")
  ax1.plot([np.ceil((i+1)*len(train_data)/batch_size) for i in range(len(val_loss_log))], val_loss_log,c="red", label="Val. Loss")
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.set_ylim(-0.01,3)
  
  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  
  color = 'tab:blue'
  ax2.plot(range(len(acc_log)), acc_log, c=color, label="Train Acc.", alpha=0.25)
  ax2.plot([np.ceil((i+1)*len(train_data)/batch_size) for i in range(len(val_acc_log))], val_acc_log,c="blue", label="Val. Acc.")
  ax2.set_ylabel(" Accuracy", c=color)
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_ylim(-0.01,1.01)
  
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  ax1.legend(loc="center")
  ax2.legend(loc="center right")
  plt.show()


if __name__=="__main__":
  main()
