# Variable <--> Tensor <--> numpy in pytorch

# Torch 自称为神经网络界的 Numpy, 因为他能将 torch 产生的 tensor 放在 GPU 中加速运算
# (前提是你有合适的 GPU), 就像 Numpy 会把 array 放在 CPU 中加速运算.

# PyTorch 使用的技术为自动微分（automatic differentiation）。
# 在这种机制下，系统会有一个 Recorder 来记录我们执行的运算，然后再反向计算对应的梯度。
# 这种技术在构建神经网络的过程中十分强大，因为我们可以通过计算前向传播过程中参数的微分来节省时间。

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200) # data (tensor), shape=(100, 1)
x = Variable(x)    # transform to Variable
x_np = x.data.numpy()   # to plot the data format in torch can't be recognized by matplotlib. In order to plot, transforming to numpy data necessary

# activation function
y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

# y_softmax = F.softmax(x)
# softmax is a special kind of activation function, it is about probability
# and will make the sum as 1.

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()



np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)  # transform np data to torch data, give numpy data (np_data) in form of ndarray  (n-dimension array)
tensor2array = torch_data.numpy()  # transform tensor to numpy

print(
    '\nnumpy array:', np_data,
    '\ntorch tensor:', torch_data,
    '\ntensor to array:', tensor2array,
)



'''math calculation in torch'''
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor


# abs 绝对值计算
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),          # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor)      # [1 2 1 2]
)

# sin   三角函数 sin
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

# mean  均值
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),         # 0.0
    '\ntorch: ', torch.mean(tensor)     # 0.0
)


#除了简单的计算, 矩阵运算才是神经网络中最重要的部分. 所以我们展示下矩阵的乘法. 注意一下包含了一个 numpy 中可行, 但是 torch 中不可行的方式.
# matrix multiplication 矩阵点乘
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor


# matrix-dot-multiplication
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
)

# multiplication. multiply the corresponding positions
print(
    '\ntorch: ', torch.mul(tensor, tensor)   # torch 会转换成 [1,2,3,4].dot([1,2,3,4) = 30.0
)




# Variable can back-propagation, Tensor can't BP
tensor = torch.FloatTensor([[1, 2], [3, 4]])   # define FloatTensor by using torch.FloatTensor()
variable = Variable(tensor, requires_grad=True)  # 把鸡蛋放到篮子里. requires_grad: build grad graph

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

# back-propagation means deviation back propagate
v_out.backward()
print(variable.grad)  # after v_out.backward(), check the grad of variable
print(variable.creator)



print(variable)
print(variable.data)   # data in Variable is in form of Tensor. variable is in form of Variable
print(variable.data.numpy())   # transform variable.data (in form of tensor) to the form of numpy






