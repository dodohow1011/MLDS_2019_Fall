import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

PI = math.pi
device = torch.device("cpu")
sz = 9

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

def sample_weight(parameters):
    for p in parameters:
        sz = list(p.size())
        if len(sz) == 2:
            for i in range(sz[0]):
                for j in range(sz[1]):
                    p[i, j].data *= 1 + 0.001 * (np.random.rand() - 0.5)
        else:
            for i in range(sz[0]):
                p[i].data *= 1 + 0.001 * (np.random.rand() - 0.5)

def main():
    ##########################
    ## sin(3*pi*x)/(3*pi*x) ##
    ##########################
    x = np.arange(0,1,0.005, dtype='f').reshape((-1,1))
    y = np.sinc(3*x)
    train_x, train_y = torch.from_numpy(x), torch.from_numpy(y)
    train_x, tain_y = Variable(train_x, requires_grad = True), Variable(train_y, requires_grad = True)

    minimal_ratio_all = []
    loss_all = []

    model = Net()
    loss_func = nn.MSELoss()

    for epoch in range(100):
        model.load_state_dict(torch.load('./models/gradient_norm_para_{}.pkl'.format(epoch)))
        prediction = model(train_x)
        best_loss = loss_func(prediction, train_y)

        # minimal ratio
        minimal_count = 0
        sample_num = 1000
        for sample_index in range(sample_num):
            model.load_state_dict(torch.load('./models/gradient_norm_para_{}.pkl'.format(epoch)))
            sample_weight(model.parameters())
            prediction = model(train_x)
            loss = loss_func(prediction, train_y)
            if best_loss.data.cpu().numpy() < loss.data.cpu().numpy():
                minimal_count = minimal_count + 1

        minimal_ratio = minimal_count / sample_num
        minimal_ratio_all.append(minimal_ratio)
        loss_all.append(best_loss.data.cpu().numpy())

        print('epoch: %3d, loss: %.6f, minimal_ratio: %.6f' %(epoch, best_loss.data.cpu().numpy(), minimal_ratio))

    plt.scatter(np.array(minimal_ratio_all), np.array(loss_all))
    plt.title('minimal_loss vs loss')
    plt.xlabel('minimal_ratio')
    plt.ylabel('loss')
    plt.savefig('minimal_ratio_vs_loss.png')


if __name__ == '__main__':
    main()

