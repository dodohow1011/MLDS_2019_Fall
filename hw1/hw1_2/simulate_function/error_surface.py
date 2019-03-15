import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

PI = math.pi
device = torch.device("cpu")
sz = 9
sample_num = 50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, sz)
        self.fc2 = nn.Linear(sz, sz)
        self.fc3 = nn.Linear(sz, sz)
        self.fc4 = nn.Linear(sz, sz)
        self.fc5 = nn.Linear(sz, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def gradient_n(parameters):
    grad_all = 0.0
    for p in parameters:
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all = grad + grad_all
    return grad_all ** 0.5

def _tsne(parameters):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim = tsne.fit_transform(parameters)
    X, Y = low_dim[:, 0], low_dim[:, 1]

    return X, Y

def flatten_param(parameters):
    all_parameters = []
    for p in parameters:
        all_parameters += list(p.data.numpy().flatten())
    return np.array(all_parameters)

def main():
    ##########################
    ## sin(3*pi*x)/(3*pi*x) ##
    ##########################
    x = np.arange(0,1,0.005, dtype='f').reshape((-1,1))
    y = np.sinc(3*x)
    train_x, train_y = torch.from_numpy(x), torch.from_numpy(y)
    train_x, tain_y = Variable(train_x, requires_grad = True), Variable(train_y, requires_grad = True)

    loss_all = []
    training_point = []
    sample_point = []
    dim_x = []
    dim_y = []

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    optimizer_gradient_norm = optim.Adam(model.parameters(), lr=0.0005)
    loss_func = nn.MSELoss()

    for epoch in range(50):
        prediction = model(train_x)
        loss = loss_func(prediction, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all.append(loss.data.cpu().numpy())
        #print('epoch: %03d, loss: %.6f' %(epoch, loss), end='\r')

        training_point.append(flatten_param(model.parameters()))
    
    gradient_norm = np.zeros([1,1])

    for epoch in range(50,80):
        prediction = model(train_x)
        
        gradient_norm[0][0] = gradient_n(model.parameters())
        gradient_Norm = Variable(torch.from_numpy(gradient_norm), requires_grad = True)
        zero_gdn = torch.zeros_like(gradient_Norm)
        optimizer_gradient_norm.zero_grad()

        loss = loss_func(gradient_Norm, zero_gdn)
        loss.backward()
        optimizer_gradient_norm.step()

        #print('epoch: %03d, loss: %.6f' %(epoch, gradient_norm[0][0]), end='\r')

        loss = loss_func(prediction, train_y)
        loss_all.append(loss.data.cpu().numpy())
        training_point.append(flatten_param(model.parameters()))

    loss_all = np.array(loss_all)
    training_point = np.array(training_point)
    _x, _y = _tsne(training_point)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(xs=_x, ys=_y, zs=loss_all, zdir='z', label='ys=0, zdir=z')
    ax.text(_x[-1], _y[-1], loss_all[-1], 'END', color='black', fontsize=10) 
    plt.savefig('training_point_3D.png')





    '''    
    plt.scatter(x, result)
    plt.title('prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('test.png')
    '''

if __name__ == '__main__':
    main()