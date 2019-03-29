import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
## load mnist dataset

#batch_size = [10, 20, 30, 60, 80, 100, 200, 300, 600, 800, 1000, 2000, 3000, 6000, 8000, 10000, 20000, 30000, 60000]
batch_size = [10, 20, 30, 60, 80, 100, 200, 300, 600, 800, 1000]

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
use_cuda = torch.cuda.is_available()
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

## network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

## training
model = LeNet()

if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.CrossEntropyLoss()

acc_array = []

for batch_size_index in range(len(batch_size)):
    print('now_batch_size:%03d.' %batch_size[batch_size_index])
    ave_loss = 0
    data_loader = torch.utils.data.DataLoader(
        dataset=dset.MNIST(root=root, train=True, transform=trans, download=True),
        batch_size=batch_size[batch_size_index],
        shuffle=True)
    model.load_state_dict(torch.load('./models/mnist_batch_size_{}.pkl'.format(batch_size[batch_size_index])))
    for batch_idx, (x, target) in enumerate(data_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target)
        ave_loss = ave_loss * 0.9 + loss.data.cpu().numpy() * 0.1
        optimizer.step()
    acc_array.append(1-ave_loss)

np.save('train_acc.npy', acc_array)