# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 19:12:16 2021

@author: BoesAn
"""
import matplotlib.pyplot as plt
import numpy as np
import logging

def main():
  X_train, Y_train, X_val, Y_val, X_test = loadData()
  for i in X_test:
    displayExample(i)
  
  
def displayExample(x):
  plt.imshow(x.reshape(28,28),cmap="gray")
  plt.show()


def loadData(normalize = True):
  train = np.loadtxt("mnist_small_train.csv", delimiter=",", dtype=np.float64)
  val = np.loadtxt("mnist_small_val.csv", delimiter=",", dtype=np.float64)
  test = np.loadtxt("mnist_small_test.csv", delimiter=",", dtype=np.float64)

  # Normalize Our Data
  if normalize:
    X_train = train[:,:-1]/256-0.5
    X_val = val[:,:-1]/256-0.5
    X_test = test/256-0.5
  else:
    X_train = train[:,:-1]
    X_val = val[:,:-1]
    X_test = test

  Y_train = train[:,-1].astype(np.int)[:,np.newaxis]
  Y_val = val[:,-1].astype(np.int)[:,np.newaxis]

  logging.info("Loaded train: " + str(X_train.shape))
  logging.info("Loaded val: " + str(X_val.shape))
  logging.info("Loaded test: "+ str(X_test.shape)) 

  return X_train, Y_train, X_val, Y_val, X_test

if __name__=="__main__":
  main()