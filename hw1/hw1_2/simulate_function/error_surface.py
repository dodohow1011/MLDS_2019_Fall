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

def main():
    ##########################
    ## sin(3*pi*x)/(3*pi*x) ##
    ##########################
    x = np.arange(0,1,0.005, dtype='f').reshape((-1,1))
    y = np.sinc(3*x)
    train_x, train_y = torch.from_numpy(x), torch.from_numpy(y)
    train_x, tain_y = Variable(train_x, requires_grad = True), Variable(train_y, requires_grad = True)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    optimizer_gradient_norm = optim.Adam(model.parameters(), lr=0.0005)
    loss_func = nn.MSELoss()

    for epoch in range(100):
        prediction = model(train_x)
        loss = loss_func(prediction, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('epoch: %03d, loss: %.6f' %(epoch, loss), end='\r')

        #training_point.append(flatten_param(model.parameters()))
        #loss_all.append(loss.data.cpu().numpy())

        torch.save(model.state_dict(), './models/error_surface_para_{}.pkl'.format(epoch))

    

if __name__ == '__main__':
    main()