from turtle import width
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
font = {'weight' : 'normal','size'   : 22}
matplotlib.rc('font', **font)
import logging
from datetime import datetime
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')



######################################################
# Q4 Implement Init, Forward, and Backward For Layers
######################################################
def simple_softmax(x):
  eList = np.exp(x)
  return eList / np.sum(eList)

def softmaxDeepnotes(X):
  exps = np.exp(X - np.max(X))
  return exps / np.sum(exps)

def softmax(x):
  x -= np.max(x,axis=1)[:,np.newaxis]  # Numerical stability trick
  return np.exp(x) / (np.sum(np.exp(x),axis=1)[:,np.newaxis])

class CrossEntropySoftmax:
  
  # Compute the cross entropy loss after performing softmax
  # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
  # labels -- batch_size x 1 vector of integer label id (0,1,2) where labels[i] is the label for batch element i
  #
  # Output should be a positive scalar value equal to the average cross entropy loss after softmax
  
# =============================================================================
#     self.logitsSoftmax = softmax(logits)	    return -np.mean(np.log(self.probs[np.arange(len(self.probs))[:,np.newaxis],labels]+0.00001))
#     logLikelihood = -np.log(self.logitsSoftmax[range(labels.shape[0]), labels]+0.00001)	
#     return np.sum(logLikelihood)/labels.shape[0]
# =============================================================================
  
  def forward(self, logits, labels):
    self.logitsSoftmax = softmax(logits)
    self.labels = labels
    return -np.mean(np.log(self.logitsSoftmax[np.arange(len(self.logitsSoftmax))[:,np.newaxis],labels]+0.00001))

  def backward(self):
    grad = self.logitsSoftmax 
    grad[np.arange(len(self.logitsSoftmax ))[:,np.newaxis],self.labels] -=  1
    return  grad.astype(np.float64)/len(self.logitsSoftmax )
    



class ReLU:

  # Forward pass is max(0,input)
  def forward(self, input):
    self.mask = (input > 0)
    return input * self.mask
  
  # Backward pass masks out same elements
  def backward(self, grad):
    return grad * self.mask

  # No parameters so nothing to do during a gradient descent step
  def step(self,step_size, currentStep):
    return


class LinearLayer:

  # Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
  def __init__(self, input_dim, output_dim):
    self.averageGradient = 0
    self.numGradients = 0
    self.weights = np.random.randn(input_dim, output_dim)* np.sqrt(2. / input_dim)
    self.bias = np.ones( (1,output_dim) )*0.5
    self.mt = np.ones((1,output_dim))
    self.vt = np.ones((1,output_dim))

 # During the forward pass, we simply compute Xw+b
  def forward(self, input):
    self.input = input #Storing X
    return  self.input@self.weights + self.bias


  # Inputs:
  #
  # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where 
  #         the i'th row is the gradient of the loss of example i with respect 
  #         to z_i (the output of this layer for example i)

  # Computes and stores:
  #
  # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient
  #                       of the loss with respect to the weights of this layer. 
  #                       This is an summation over the gradient of the loss of
  #                       each example with respect to the weights.
  #
  # self.grad_bias dL/dZ--     A (1 x output_dim) matrix storing the gradient
  #                       of the loss with respect to the bias of this layer. 
  #                       This is an summation over the gradient of the loss of
  #                       each example with respect to the bias.
  
  # Return Value:
  #
  # grad_input dL/dX -- For a batch size of n, grad_input is a (n x input_dim) matrix where
  #               the i'th row is the gradient of the loss of example i with respect 
  #               to x_i (the input of this layer for example i) 

  def backward(self, grad): # grad is dL/dZ.
    #have dL/dZ, need dL/dX
    self.grad_weights = (self.input.T @ grad) # Compute dL/dW
    self.grad_bias = grad.sum() # Compute dL/db
    self.grad = grad
    return (grad @ self.weights.T)# Compute dL/dX
    
  ######################################################
  # Q5 Implement ADAM with Weight Decay
  ######################################################  
  def step(self, step_size, beta1 = .1, beta2 = .1, epsilon = 1e-6, currentStep = -1):
    #TODO: implment weight decay
    self.mt = beta1 * self.mt + (1. - beta1) * self.grad_weights
    self.vt = beta2 * self.vt + (1. - beta2) * self.grad_weights ** 2
    mt = self.mt
    vt = self.vt
    mt_hat = self.mt / (1. - beta1 ** (currentStep+1))
    vt_hat = self.vt / (1. - beta2 ** (currentStep+1))
    self.weights = self.weights -  (step_size / (np.sqrt(vt_hat) + epsilon)) * mt_hat
    #self.weights -= step_size*self.grad_weights
    self.bias -= step_size*self.grad_bias


######################################################
# Q6 Implement Evaluation and Training Loop
###################################################### 

# Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
  val_loss_running = 0
  val_acc_running = 0
  j=0
  lossFunc = CrossEntropySoftmax()
  while j < len(X_val):
    b = min(batch_size, len(X_val)-j)
    X_batch = X_val[j:j+b]
    Y_batch = Y_val[j:j+b].astype(int)
    logits = model.forward(X_batch)
    loss = lossFunc.forward(logits, Y_batch)
    acc = np.mean( np.argmax(logits,axis=1)[:,np.newaxis] == Y_batch)
    val_loss_running += loss*b
    val_acc_running += acc*b       
    j+=batch_size
  return val_loss_running/len(X_val), val_acc_running/len(X_val)


def main():
  # Load data
  X_train, Y_train, X_val, Y_val, X_test, Y_test = loadCIFAR10Data()
  for n in [1,.1,.01,.001,.0001]:
    runNN(X_train, Y_train, X_val, Y_val, X_test, Y_test, n)
  
def runNN(X_train, Y_train, X_val, Y_val, X_test, Y_test, param):

  # Set optimization parameters (NEED TO CHANGE THESE)
  batch_size = 200
  max_epochs = 15
  step_size = .01
  number_of_layers = 2
  width_of_layers = 8
  randomSeed = 1005


  
  # Some helpful dimensions
  num_examples, input_dim = X_train.shape
  output_dim = 3 # number of class labels


  # Build a network with input feature dimensions, output feature dimension,
  # hidden dimension, and number of layers as specified below. You can edit this as you please.



  #trainingIndexes = np.arange(len(X_train))
  # For each epoch below max epochs
  bestRunPercent = -1
  bestRunEpoch = -1
  bestRunLoss = -1
  np.random.seed(randomSeed)
  # Some lists for book-keeping for plotting later
  trainingIndexes = np.arange(len(X_train))
  bestRunPercent = -1
  bestRunEpoch = -1
  bestRunLoss = -1
  losses = []
  val_losses = []
  accs = []
  val_accs = []
  lossFunc = CrossEntropySoftmax()
  acc_running = 0
  loss_running = 0
  step_size = param
  network = FeedForwardNeuralNetwork(input_dim,output_dim, width_of_layers, number_of_layers)
  for i in range(max_epochs):
    np.random.shuffle(trainingIndexes) # Scramble order of examples
    j=acc_running=loss_running=0                
    # for each batch in data:
    while j < len(X_train):
      # Gather batch
      batchInstanceSize = min(batch_size, len(X_train)-j)
      X_batch = X_train[trainingIndexes[j:j+batchInstanceSize]]
      Y_batch = Y_train[trainingIndexes[j:j+batchInstanceSize]].astype(int)      
      results = network.forward(X_batch) # Compute forward pass
      accuracy = np.mean( np.argmax(results,axis=1)[:,np.newaxis] == Y_batch)      
      loss = lossFunc.forward(results, Y_batch) # Compute loss
      #print(loss)
      lossGrad = lossFunc.backward()
      network.backward(lossGrad) # Backward loss and networks
      network.step(step_size, currentStep=i)# Take optimizer step

      # Book-keeping for loss / accuracy
      losses.append(loss)
      accs.append(accuracy)
      loss_running += loss*batchInstanceSize
      acc_running += accuracy*batchInstanceSize

      j+=batch_size
  
    # Evaluate performance on validation.

    
    ###############################################################
    # Print some stats about the optimization process after each epoch
    ###############################################################
    vloss, vacc = evaluate(network, X_val, Y_val, batch_size)
    val_losses.append(vloss)
    val_accs.append(vacc)
    #epoch_avg_loss = loss_running/len(X_train)# -- average training loss across batches this epoch
    #epoch_avg_acc = acc_running / len(X_train)*100 #-- average accuracy across batches this epoch
    
    
    logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%  Val Acc:  {:8.4}%".format(i,loss_running/len(X_train), acc_running / len(X_train)*100,vacc*100))
    if vacc > bestRunPercent:
        bestRunPercent = vacc
        bestRunEpoch = i
        bestRunLoss = loss_running/len(X_train)
    
  ###############################################################
  # Code for producing output plot requires
  ###############################################################
  # losses -- a list of average loss per batch in training
  # accs -- a list of accuracies per batch in training
  # val_losses -- a list of average validation loss at each epoch
  # val_acc -- a list of validation accuracy at each epoch
  # batch_size -- the batch size
  ################################################################

  # Plot training and validation curves
  fig, ax1 = plt.subplots(figsize=(16,9))
  color = 'tab:red'
  ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
  ax1.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_losses))], val_losses,c="red", label="Val. Loss")
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
  ax1.tick_params(axis='y', labelcolor=color)
  #ax1.set_ylim(-0.01,3)
  
  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:blue'
  ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
  ax2.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_accs))], val_accs,c="blue", label="Val. Acc.")
  ax2.set_ylabel(" Accuracy", c=color)
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_ylim(-0.01,1.01)
  
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  ax1.legend(loc="center")
  ax2.legend(loc="center right")
  plt.show()  
  file = open("runs.txt", "a")
  runTimeKey = str(datetime.now().strftime("%Y%b%dT%H:%M:%S"))
  file.writelines("{},{},{},{},{},{},{},{},{},{}\n".format(runTimeKey,randomSeed,step_size,batch_size,max_epochs,number_of_layers,width_of_layers,bestRunEpoch,bestRunPercent*100,bestRunLoss))
  file.close()  

  ################################
  # Q7 Tune and Evaluate on Test
  ################################
  _, tacc = evaluate(network, X_test, Y_test, batch_size)
  print(tacc)



#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################

class FeedForwardNeuralNetwork:

  def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
 
    if num_layers == 1:
      self.layers = [LinearLayer(input_dim, output_dim)]
    else:
      self.layers = [LinearLayer(input_dim, hidden_dim)]
      self.layers.append(ReLU())
      for i in range(num_layers-2):
        self.layers.append(LinearLayer(hidden_dim, hidden_dim))
        self.layers.append(ReLU())
      self.layers.append(LinearLayer(hidden_dim, output_dim))

  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def backward(self, grad):
    for layer in reversed(self.layers):
      grad = layer.backward(grad)

  def step(self, step_size, currentStep):
    for layer in self.layers:
      layer.step(step_size, currentStep)





#####################################################
# Utility Functions for Loading and Visualizing Data
#####################################################

def loadCIFAR10Data():

  with open("cifar10_hst_train", 'rb') as fo:
    data = pickle.load(fo)
  X_train = data['images']
  Y_train = data['labels']

  with open("cifar10_hst_val", 'rb') as fo:
    data = pickle.load(fo)
  X_val = data['images']
  Y_val = data['labels']

  with open("cifar10_hst_test", 'rb') as fo:
    data = pickle.load(fo)
  X_test = data['images']
  Y_test = data['labels']
  
  logging.info("Loaded train: " + str(X_train.shape))
  logging.info("Loaded val: " + str(X_val.shape))
  logging.info("Loaded test: " + str(X_test.shape))
  X_train = (X_train/256)
  return X_train, Y_train, X_val, Y_val, X_test, Y_test


def displayExample(x):
  r = x[:1024].reshape(32,32)
  g = x[1024:2048].reshape(32,32)
  b = x[2048:].reshape(32,32)
  
  plt.imshow(np.stack([r,g,b],axis=2))
  plt.axis('off')
  plt.show()


if __name__=="__main__":
  main()
