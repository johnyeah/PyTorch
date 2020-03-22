# PyTorch Regression
# Morvan Python

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable


''' creat data '''
torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  #in torch, only 2-dimentions can be processed. torch.unsqueeze() add another dim, dim= means which dim be added
y = x.pow(2) + 0.2*torch.rand(x.size())


plt.figure()
plt.scatter(x.data.numpy(), y.data.numpy())
plt.grid(True)
plt.draw()



### transform x, y to type Variable, because only typle Varibale can be inputed in NN
#x, y = Variable(x), Variable(y)

#plt.scatter(x.data.numpy(), y.data.numpy())
#plt.show()


''' build NN model structure '''
class Net(torch.nn.Module):   # define neural network that inherits the module (torch.nn.Module) form torch
    def __init__(self, n_features, n_hidden, n_output):   # required information for building layers
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):  # forward process of NN
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


''' define the number of neurons in each layers '''
net = Net(1, 10, 1)
print(net)




''' optimize function '''
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # MSELoss: Mean Square Error for regression problem



''' training model '''
plt.figure()
plt.ion()   # Turn the interactive mode on.

for t in range(200):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0:   # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  # lw: line width
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)


plt.ioff() # Turn the interactive mode off.
plt.show()





torch.save(net, 'net.pkl')  # 保存整个网络
# torch.save(net.state_dict(), 'net_params.pkl')  # 只保存网络中的参数 (速度快, 占内存少)


# 提取网络
#net2 = torch.load('net.pkl')  #只提取网络参数

# net3.load_state_dict(torch.load('net_params.pkl'))  # 将保存的参数复制到 net3

