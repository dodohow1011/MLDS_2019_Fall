import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from collections import OrderedDict

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

def generate_state_dict(vector, model):
    return_orderdict = OrderedDict()
    dummy_state_dict = model.state_dict()
    cum_idx = 0
    for key, value in dummy_state_dict.items():
        sz = np.array(value.size())
        length = np.prod(sz)
        weight = vector[cum_idx:cum_idx + length].reshape(sz)
        return_orderdict[key] = torch.FloatTensor(weight)
        cum_idx = cum_idx + length
    return return_orderdict

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
    #_x, _y = _tsne(np.array(all_point))
    pca = PCA(n_components=2)
    pca = pca.fit(np.array(training_point))
    low_dim = pca.transform(training_point)
    _x, _y= low_dim[:, 0],  low_dim[:, 1]
    x_min, x_max, y_min, y_max = np.amin(_x), np.amax(_x), np.amin(_y), np.amax(_y)
    #print(x_min, x_max, y_min, y_max)

    sample_num = 20
    sample_x = np.linspace(x_min, x_max, num=sample_num)
    sample_y = np.linspace(y_min, y_max, num=sample_num)
    _sample_x = []
    _sample_y = []

    for i in range(sample_num):
        for j in range(sample_num):
            tmp_vector = np.append(sample_x[i],sample_y[j])
            sample_point.append(pca.inverse_transform(tmp_vector))
            _sample_x.append(sample_x[i])
            _sample_y.append(sample_y[j])
            #print(sample_point[-1])

    for sample_index in range(len(sample_point)):
        vector_state_dict = generate_state_dict(sample_point[i], model)
        model.load_state_dict(vector_state_dict)

        prediction = model(train_x)
        loss = loss_func(prediction, train_y)
        sample_loss.append(loss.data.cpu().numpy())

    _v = np.append(_x[0],_y[0])
    t_point = pca.inverse_transform(_v)
    print(t_point)
    print(training_point[0])


    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.text(_x[0], _y[0], loss_all[0], 'START', color='black', fontsize=10)
    ax.text(_x[99], _y[99], loss_all[99], 'END', color='black', fontsize=10)
    ax.plot(_x[0:100], _y[0:100], loss_all[0:100], zdir='z', label='ys=0, zdir=z')
    ax.plot_trisurf(_sample_x, _sample_y, sample_loss, linewidth=0.2, antialiased=True, alpha=1.0, cmap='coolwarm')
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