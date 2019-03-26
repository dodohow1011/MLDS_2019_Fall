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

def gradient_n(parameters):
    grad_all = 0.0
    for p in parameters:
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all = grad + grad_all
    return grad_all ** 0.5

def main():
    ##########################
    ## sin(3*pi*x)/(3*pi*x) ##
    ##########################
    x = np.arange(0,1,0.005, dtype='f').reshape((-1,1))
    y = np.sinc(3*x)
    train_x, train_y = torch.from_numpy(x), torch.from_numpy(y)
    train_x, tain_y = Variable(train_x, requires_grad = True), Variable(train_y, requires_grad = True)

    loss_all = []
    gradient_norm_all = []

    for train_times in range(100):
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
            print('time: %03d, epoch: %03d, loss: %.6f' %(train_times, epoch, loss), end='\r')
        
        gradient_norm = np.zeros([1,1])

        for epoch in range(100,3000):
            prediction = model(train_x)
            
            gradient_norm[0][0] = gradient_n(model.parameters())
            gradient_Norm = Variable(torch.from_numpy(gradient_norm), requires_grad = True)
            zero_gdn = torch.zeros_like(gradient_Norm)
            optimizer_gradient_norm.zero_grad()

            loss = loss_func(gradient_Norm, zero_gdn)
            loss.backward()
            optimizer_gradient_norm.step()

            print('time: %03d, epoch: %03d, loss: %.6f' %(train_times, epoch, gradient_norm[0][0]), end='\r')

            '''
            if epoch == 1499:
                result = prediction.detach().numpy()
            '''
        torch.save(model.state_dict(), './models/gradient_norm_para_{}.pkl'.format(train_times))

    '''    
    plt.scatter(x, result)
    plt.title('prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('test.png')
    '''

if __name__ == '__main__':
    main()