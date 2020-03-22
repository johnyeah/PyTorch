'''
batch normalization and compare with no batch normalization
own dataset, use morvan's model

'''


import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


# torch.manual_seed(1)    # reproducible
# np.random.seed(1)

# Hyper parameters
BATCH_SIZE = 256
EPOCH = 5
LR = 0.03  # learning rate
N_HIDDEN = 8  # number of hidden layers
N_HIDDEN_NEURON = 1024  # number of neuron
N_INPUT_FEATURES = 15  # number of input features
ACTIVATION = torch.relu
B_INIT = -0.2   # use a bad bias constant initializer



'''data preprocessing'''

FILE_PATH_TRAIN = '../Old/191217_test_record_iter_fix_static1.csv'
FILE_PATH_TEST = FILE_PATH_TRAIN     #'sample_data/100_samples.csv'

input_features = ['grip_pos_x', 'grip_pos_y', 'grip_pos_z',
                      'grip_vel_x', 'grip_vel_y', 'grip_vel_z',
                      'obst_pos_x_1', 'obst_pos_y_1', 'obst_pos_z_1',
                      'goal_pos_x', 'goal_pos_y', 'goal_pos_z',
                      'action_x', 'action_y', 'action_z']
output_features = ['safe', 'unsafe']
label_feature = ['is_safe_action']


origin_data = pd.read_csv(FILE_PATH_TRAIN)


# number of safe,unsafe

print('num_safe_state:', np.sum(origin_data[label_feature]==True))
print('num_unsafe_state:', np.sum(origin_data[label_feature]==False))
print('safe_state_rate:', np.sum(origin_data[label_feature]==True) / len(origin_data))


# select safe_state
safe_state = origin_data.loc[origin_data['is_safe_action']==True]
safe_state = safe_state.sample(n=612095)  # randomly sample

unsafe_state = origin_data.loc[origin_data['is_safe_action']==False]

prepocessed_data = pd.concat([safe_state, unsafe_state], axis=0)
prepocessed_data = prepocessed_data.sample(frac=1)


print('num_prepocessed_data:', len(prepocessed_data))



# train dataset
train = prepocessed_data.iloc[10000:20000]
train_x = train.loc[:, input_features]
train_y = train[label_feature].astype(int)

train_x, train_y = torch.from_numpy(train_x.values).float(), torch.from_numpy(train_y.values).float()

train_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)
print('train dataset:', train_x.shape)


# test dataset
test = prepocessed_data.iloc[1000:2000]
test_x = test.loc[:, input_features]
test_y = test['is_safe_action'].astype(int)


test_x, test_y = torch.from_numpy(test_x.values).float(), torch.from_numpy(test_y.values).float()
print('test dataset:', test_x.shape)



class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []
        self.bns = []
        self.bn_input = nn.BatchNorm1d(15, momentum=0.5)   # for input data

        for i in range(N_HIDDEN):               # build hidden layers and BN layers
            input_size = N_INPUT_FEATURES if i == 0 else N_HIDDEN_NEURON
            fc = nn.Linear(input_size, N_HIDDEN_NEURON)
            setattr(self, 'fc%i' % i, fc)       # IMPORTANT set layer to the Module
            self._set_init(fc)                  # parameters initialization
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(N_HIDDEN_NEURON, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)   # IMPORTANT set layer to the Module
                self.bns.append(bn)

        self.predict = nn.Linear(N_HIDDEN_NEURON, 1)         # output layer
        self._set_init(self.predict)            # parameters initialization

    def _set_init(self, layer):
        init.normal_(layer.weight, mean=0., std=.1)
        init.constant_(layer.bias, B_INIT)

    def forward(self, x):
        pre_activation = [x]
        if self.do_bn: x = self.bn_input(x)     # input batch normalization
        layer_input = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            pre_activation.append(x)
            if self.do_bn: x = self.bns[i](x)   # batch normalization
            x = ACTIVATION(x)
            layer_input.append(x)
        out = self.predict(x)
        return out, layer_input, pre_activation

nets = [Net(batch_normalization=False), Net(batch_normalization=True)]

print(nets)

# print(*nets)    # print net architecture

opts = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]
loss_func = torch.nn.MSELoss()



def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0:
            p_range = (-1, 1); the_range = (-1, 1)
        else:
            p_range = (-1, 1); the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5);ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359');ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]: a.set_yticks(());a.set_xticks(())
        ax_pa_bn.set_xticks(p_range);ax_bn.set_xticks(the_range)
        axs[0, 0].set_ylabel('PreAct');axs[1, 0].set_ylabel('BN PreAct');axs[2, 0].set_ylabel('Act');axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)


if __name__ == "__main__":
    f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))
    plt.ion()  # something about plotting
    plt.show()

    # training
    losses = [[], []]  # recode loss for two networks

    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        layer_inputs, pre_acts = [], []
        for step, (b_x, b_y) in enumerate(train_loader):
            for net, opt in zip(nets, opts):     # train for each network
                pred, _, _ = net(b_x)
                loss = loss_func(pred, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()    # it will also learns the parameters in Batch Normalization

        for net, l in zip(nets, losses):
            net.eval()              # set eval mode to fix moving_mean and moving_var
            pred, layer_input, pre_act = net(test_x)
            l.append(loss_func(pred, test_y).data.item())
            layer_inputs.append(layer_input)
            pre_acts.append(pre_act)
            net.train()             # free moving_mean and moving_var
        plot_histogram(*layer_inputs, *pre_acts)     # plot histogram



    plt.ioff()

    # plot training loss
    plt.figure(2)
    plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
    plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
    plt.xlabel('step')
    plt.ylabel('test loss')
    plt.legend(loc='best')
    plt.show()


'''
    # evaluation
    # set net to eval mode to freeze the parameters in batch normalization layers
    [net.eval() for net in nets]    # set eval mode to fix moving_mean and moving_var
    preds = [net(test_x)[0] for net in nets]
    plt.figure(3)
    plt.plot(test_x.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
    plt.plot(test_x.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
    plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
    plt.legend(loc='best')
'''


