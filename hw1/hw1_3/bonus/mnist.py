import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import argparse
## load mnist dataset

parser = argparse.ArgumentParser(description='hw1-3-3 sharpness plot')
parser.add_argument('-b', '--batch', default=50, type=int, help='batch size to computing loss (default: 128)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

batch_size = args.batch

data_loader = torch.utils.data.DataLoader(
                 dataset=dset.MNIST(root=root, train=True, transform=trans, download=True),
                 batch_size=batch_size,
                 shuffle=True)

print('==>>> total trainning batch number: {}'.format(len(data_loader)))

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

if args.batch < 90:
    epochs = 10
else:
    epochs = 30

for epoch in range(epochs):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(data_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target)
        ave_loss = ave_loss * 0.9 + loss.data.cpu().numpy() * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(data_loader):
            print ('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss))

torch.save(model.state_dict(), './models/mnist_batch_size_{}.pkl'.format(args.batch))
  
