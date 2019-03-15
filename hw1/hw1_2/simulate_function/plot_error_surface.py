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
from matplotlib import cm

PI = math.pi
device = torch.device("cpu")
sz = 9
sample_num = 20

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

def sample_weight(parameters):
    for p in parameters:
        sz = list(p.size())
        if len(sz) == 2:
            for i in range(sz[0]):
                for j in range(sz[1]):
                    p[i, j].data *= 1 + 0.000001 * (np.random.rand() - 0.5)
        else:
            for i in range(sz[0]):
                p[i].data *= 1 + 0.000001 * (np.random.rand() - 0.5)

def main():
    ##########################
    ## sin(3*pi*x)/(3*pi*x) ##
    ##########################
    x = np.arange(0,1,0.005, dtype='f').reshape((-1,1))
    y = np.sinc(3*x)
    train_x, train_y = torch.from_numpy(x), torch.from_numpy(y)
    train_x, tain_y = Variable(train_x, requires_grad = True), Variable(train_y, requires_grad = True)

    training_loss = []
    sample_loss = []
    training_point = []
    sample_point = []
    all_point = []

    model = Net()
    loss_func = nn.MSELoss()

    for epoch in range(100):

        model.load_state_dict(torch.load('./models/error_surface_para_{}.pkl'.format(epoch)))
        prediction = model(train_x)
        loss = loss_func(prediction, train_y)
        training_loss.append(loss.data.cpu().numpy())
        training_point.append(flatten_param(model.parameters()))
        print('epoch: %03d, loss: %.6f' %(epoch, loss), end='\r')
            	
    
    loss_all = training_loss
    all_point = training_point
    _x, _y = _tsne(np.array(all_point))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.text(_x[0], _y[0], loss_all[0], 'START', color='black', fontsize=10)
    ax.text(_x[9], _y[9], loss_all[9], 'END', color='black', fontsize=10)
    ax.plot(_x[0:10], _y[0:10], loss_all[0:10], zdir='z', label='ys=0, zdir=z')
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