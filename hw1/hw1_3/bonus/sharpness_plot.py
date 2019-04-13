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
import matplotlib.pyplot as plt

#batch_size = [10, 20, 30, 60, 80, 100, 200, 300, 600, 800, 1000, 2000, 3000, 6000, 8000, 10000, 20000, 30000, 60000]
batch_size = [10, 20, 30, 60, 80, 100, 200, 300, 600, 800, 1000]
sample_times = 1000
epsilon = 1e-4

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
use_cuda = torch.cuda.is_available()
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

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

def sample_weight(parameters):
    for p in parameters:
        sz = list(p.size())
        if len(sz) == 2:
            for i in range(sz[0]):
                for j in range(sz[1]):
                    p[i, j].data += (np.random.random() - 0.5) * epsilon
        else:
            for i in range(sz[0]):
                p[i].data += (np.random.random() - 0.5) * epsilon

def main():
	model = LeNet()
	sample_model = LeNet()

	if use_cuda:
		model = model.cuda()

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	criterion = nn.CrossEntropyLoss()


	sharpness_array = []
	test_acc = []

	for batch_size_index in range(len(batch_size)):

		data_loader = torch.utils.data.DataLoader(
			dataset=dset.MNIST(root=root, train=False, transform=trans, download=True),
			batch_size=10000,
			shuffle=True)

		model.load_state_dict(torch.load('./models/mnist_batch_size_{}.pkl'.format(batch_size[batch_size_index])))
		for batch_idx, (x, target) in enumerate(data_loader):
			optimizer.zero_grad()
			if use_cuda:
				x, target = x.cuda(), target.cuda()
				model.cuda()
			out = model(x)
			loss = criterion(out, target)
			L0_loss = loss.data.cpu().numpy()

			max_sharpness = float("-inf")
			for times in range(sample_times):
				print('batch_size:%03d, sample_times:%03d.' %(batch_size[batch_size_index], times))
				optimizer.zero_grad()
				sample_model.load_state_dict(torch.load('./models/mnist_batch_size_{}.pkl'.format(batch_size[batch_size_index])))
				sample_weight(sample_model.parameters())
				if use_cuda:
					sample_model.cuda()
				out = sample_model(x)
				loss = criterion(out, target)
				L1_loss = loss.data.cpu().numpy()

				sharpness = (L1_loss-L0_loss)/(1+L0_loss)
				if sharpness > max_sharpness:
					max_sharpness = sharpness
			sharpness_array.append(max_sharpness)
			test_acc.append(1-L0_loss)
			print(max_sharpness)

	print(sharpness_array)

	train_acc = np.load('train_acc.npy',)

	fig, ax1 = plt.subplots()
	color = 'tab:blue'
	ax1.set_xlabel('batch_size (log scale)')
	ax1.set_ylabel('acc', color=color)
	ax1.set_xscale("log", nonposx='clip')
	ax1.plot(batch_size, train_acc, 'b')
	ax1.plot(batch_size, test_acc, 'b--')
	ax1.tick_params(axis='y', labelcolor=color)
	ax1.legend(['train_acc', 'test_acc'], loc='upper left')

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:red'
	ax2.set_ylabel('sharpness', color=color)  # we already handled the x-label with ax1
	ax2.plot(batch_size, sharpness_array, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.savefig('sharpness.png')

if __name__ == '__main__':
	main()