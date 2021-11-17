# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 07:49:01 2021

@author: BoesAn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import  rotate

train = np.loadtxt("mnist_small_train.csv", delimiter=",", dtype=np.int)
y= train[:,-1].astype(np.int)[:,np.newaxis]
train = train[:,:-1]
minus10 = []
plus10 = []
up1 = []
down1 = []
left = []
right = []
for x in train:
    data = np.reshape(x, (28,28))
    data = rotate(data, 10)
    data = data[2:30, 2:30]
    data = data.clip(min=0)
    data = np.reshape(data, (1,28*28))
    plus10.append(data)
    data = np.reshape(x, (28,28))
    data = rotate(data, -10)
    data = data[2:30, 2:30]
    data = data.clip(min=0)
    data = np.reshape(data, (1,28*28))
    minus10.append(data)
    data = np.reshape(x, (28,28))
    data = data[:,:-1] #move left
    data = np.c_[ np.zeros(28), data]
    data = np.reshape(data, (1,28*28))
    left.append(data)
    data = np.reshape(x, (28,28))
    data = data[:,1:] #move left
    data = np.c_[data, np.zeros(28)]
    data = np.reshape(data, (1,28*28))
    right.append(data)
    data = np.reshape(x, (28,28))
    data = data[:-1,:] #move down
    data = np.vstack((np.zeros(28).T, data))
    data = np.reshape(data, (1,28*28))
    down1.append(data)
    data = np.reshape(x, (28,28))
    data = data[1:,:] #move up
    data = np.vstack((data, np.zeros(28).T))
    data = np.reshape(data, (1,28*28))
    up1.append(data)
    
rows, cols = train.shape
np.savetxt("up1.csv", np.hstack((np.array(up1).reshape(rows,cols), y)), delimiter=",",fmt='%i')
np.savetxt("down1.csv", np.hstack((np.array(down1).reshape(rows,cols), y)), delimiter=",",fmt='%i')
np.savetxt("left.csv", np.hstack((np.array(left).reshape(rows,cols), y)), delimiter=",",fmt='%i')
np.savetxt("right.csv", np.hstack((np.array(right).reshape(rows,cols), y)), delimiter=",",fmt='%i')
np.savetxt("minus10.csv", np.hstack((np.array(minus10).reshape(rows,cols), y)), delimiter=",",fmt='%i')
np.savetxt("plus10.csv", np.hstack((np.array(plus10).reshape(rows,cols), y)), delimiter=",",fmt='%i')
np.savetxt("orig.csv", np.hstack((train, y)), delimiter=",",fmt='%i')
#copy *.csv ..\allJiggles.csv



# =============================================================================
# data2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,222,225,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,147,234,252,176,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,197,253,252,208,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,178,252,253,117,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,57,252,252,253,89,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,222,253,253,79,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131,252,179,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,198,246,220,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,79,253,252,135,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,140,253,252,118,0,0,0,0,111,140,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,191,255,253,56,0,0,114,113,222,253,253,255,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,76,252,253,223,37,0,48,174,252,252,242,214,253,199,31,0,0,0,0,0,0,0,0,0,0,0,0,13,109,252,228,130,0,38,165,253,233,164,49,63,253,214,31,0,0,0,0,0,0,0,0,0,0,0,0,73,252,252,126,0,23,178,252,240,148,7,44,215,240,148,0,0,0,0,0,0,0,0,0,0,0,0,0,119,252,252,0,0,197,252,252,63,0,57,252,252,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,135,253,174,0,48,229,253,112,0,38,222,253,112,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,135,252,173,0,48,227,252,158,226,234,201,27,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,57,252,252,57,104,240,252,252,253,233,74,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,242,252,253,252,252,252,252,240,148,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,189,253,252,252,157,112,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# #plt.imshow(x.reshape(28,28),cmap="gray")
# data = np.reshape(data2, (28,28))
# plt.imshow(data)
# plt.show()
# data = data[:-1,:] #move down
# data = np.vstack((np.zeros(28).T, data))
# plt.imshow(data)
# plt.show()
# data = np.reshape(data2, (28,28))
# data = data[1:,:] #move up
# data = np.vstack((data, np.zeros(28).T))
# plt.imshow(data)
# plt.show()
# =============================================================================

# =============================================================================
# train = np.loadtxt("plus10.csv", delimiter=",", dtype=np.int)
# train = train[:10,:]
# for x in train:
#     plt.imshow(x.reshape(28,28),cmap="gray")
#     plt.show()
# =============================================================================
# =============================================================================
# print(data.shape)
# data = np.reshape(data, (1,28*28))
# print(data.shape)
# =============================================================================
