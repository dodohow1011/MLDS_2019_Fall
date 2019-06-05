from simple_CGAN_v2 import Generator
import pickle
import os
import torch
import numpy as np
from argparse import ArgumentParser
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

parser = ArgumentParser()
parser.add_argument('--model', '-m', required=True, help='Generator model')
parser.add_argument('--output', '-o', required=True, help='Output Directory')
parser.add_argument('--attribute', '-a', required=True, help='The attribute dictionary')
parser.add_argument('--attr1', required=True, help='first attribute')
parser.add_argument('--attr2', required=True, help='second attribute')
args = parser.parse_args()

checkpoint = args.model
outdir     = args.output
attr       = args.attribute
attr1      = ' '.join(args.attr1.split('_'))
attr2      = ' '.join(args.attr2.split('_'))


with open(attr, 'rb') as f:
    ATTR = pickle.load(f)

def ToEmbed(a1, a2):
    vec = np.zeros(len(ATTR))
    vec[ATTR[a1]] = 1
    vec[ATTR[a2]] = 1
    return vec.reshape((-1, 1, 1))

G = Generator(len(ATTR))
G.load_state_dict(state_dict=torch.load(checkpoint))
G.cuda()

y = list()
for i in range(25):
    y.append(ToEmbed(attr1, attr2))
y = np.array(y)

y = Variable(torch.FloatTensor(y)).cuda()
z = torch.randn(25, 100, 1, 1).cuda()

images = G(z, y)

image_list = []
for img in images:
    img = (img.data - img.data.min()) / (img.data.max() - img.data.min())
    image_list.append(img)

    epoch = checkpoint.split('-')[1][1:4]
    haha = os.path.join(outdir, '_'.join(attr1.split())+'_'+'_'.join(attr2.split())+'_epoch-'+epoch+'-step0001.png')
    save_image(make_grid(image_list, nrow=5), haha)
