# PyTorch Classification
# Morvan Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F




# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2); std: standard deviation
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()         # class0 y data (tensor), shape=(100, 1)x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floatingy = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer


''' build NN model structure '''
class Net(torch.nn.Module):   # define neural network that inherits the module (torch.nn.Module) form torch
    def __init__(self, n_features, n_hidden, n_output):   # required information for building layers
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):  # forward process of NN
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


''' define the number of neurons in each layers '''
#net = Net(2, 10, 1)
net = Net(n_features=2, n_hidden=10, n_output=2)
print(net)




'''Method2: build NN architecture'''
'''
net2 = torch.nn.sequentical(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

print(net2)   # print net2 layers

'''






''' optimize function '''
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)


# Cross Entropy for classification, calculates the softmax (probability)
# e.g. label: [0, 0, 1], calculate value: [0.1, 0.2, 0.7]
loss_func = torch.nn.CrossEntropyLoss()



''' training model '''

plt.ion()   # Turn the interactive mode on.

for t in range(100):
    out = net(x)  # the value of out likes [-2, -.12, 20] by F.softmax(out) to [0.1, 0.2, 0.7]
    loss = loss_func(out, y)

    optimizer.zero_grad() # clear gradients for next train
    loss.backward()       # backpropagation, compute gradients
    optimizer.step()      # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]   # using softmax() transforms the out to probabilistic value. perform as a activative function
       # prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy()    # torch.max() outputs the maximum, which is indexed as [1]
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()



