# PyTorch_mini-batch
# Morvan Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data




x = torch.linspace(1, 10, 10)   # x data (torch tensor)
y = torch.linspace(10, 1, 10)   # y data (torch tensor)


batch_size = 5
torch_dataset = Data.TensorDataset(x, y)




loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,   # multiprocessing to read data
)

for epoch in range(10):   # train entire dataset 3 times
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        # train your data...
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())





