from modified_CGAN_v2 import Generator, Discriminator, NOISE_DIM
from argparse import ArgumentParser
from random import shuffle, choice
from imageio import imread
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
import sys
from torch import optim, nn
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import torch

parser = ArgumentParser()
parser.add_argument('--batch', '-b', required=True, type=int, help='batch size')
parser.add_argument('--epoch', '-e', required=True, type=int, help='epoch')
parser.add_argument('--output', '-o', required=True, help='output directory')
parser.add_argument('--learn', '-l', required=True, type=float, help='learning rate')
parser.add_argument('--data', '-d', required=True, help='the training.txt file')
parser.add_argument('--interval', '-i', required=True, type=int, help='the interval between two saved models')
args = parser.parse_args()

BATCH_SIZE    = args.batch
EPOCH         = args.epoch
OUTDIR        = args.output
INTERVAL      = args.interval
RESULT        = os.path.join(OUTDIR, 'images')
LEARNING_RATE = args.learn
DATA          = []
ATTR          = {}
SAMPLE_NUM    = 25
SAVED_IMAGES  = []
MAX_SAVE      = 30
if not os.path.exists(RESULT): os.mkdir(RESULT)

print ('Saving model and history every {} epoch'.format(INTERVAL))
print ('Learning rate {}'.format(LEARNING_RATE))
print ('Batch size {}'.format(BATCH_SIZE))
print ('Epoch {}'.format(EPOCH))

# read the training data
with open(args.data, 'r') as f:
    currentID = 0
    for line in f:
        tmp = tuple(line.strip().split(','))
        DATA.append(tmp)
        if tmp[1] not in ATTR:
            ATTR[tmp[1]] = currentID
            currentID += 1
        if tmp[2] not in ATTR:
            ATTR[tmp[2]] = currentID
            currentID += 1

# save attribute dict
with open(os.path.join(OUTDIR, 'ATTR.pkl'), 'wb') as f:
    pickle.dump(ATTR, f, protocol=pickle.HIGHEST_PROTOCOL)

def ToEmbed(attr1, attr2):
    vec = np.zeros(len(ATTR))
    vec[ATTR[attr1]] = 1
    vec[ATTR[attr2]] = 1
    return vec.reshape((-1, 1, 1))

def random_noise():
    return torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).cuda()

# transform to [-1, 1]
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

def loader(data):
    batch_count = len(data) // BATCH_SIZE
    while True:
        shuffle(data)
        for batch in range(batch_count):
            real_img = list()
            real_y   = list()
            fake_y   = list()
            for i in range(BATCH_SIZE):
                pth   = data[batch*BATCH_SIZE+i][0]
                attr1 = data[batch*BATCH_SIZE+i][1]
                attr2 = data[batch*BATCH_SIZE+i][2]
                real_img.append(transform(np.array(imread(pth))).tolist())
                real_y.append(ToEmbed(attr1, attr2))
                fake_attr1 = str(attr1)
                fake_attr2 = str(attr2)
                while (fake_attr1 == attr1 and fake_attr2 == attr2) or (not fake_attr1.endswith('hair') or not fake_attr2.endswith('eyes')):
                    fake_attr1 = choice(list(ATTR))
                    fake_attr2 = choice(list(ATTR))
                fake_y.append(ToEmbed(fake_attr1, fake_attr2))
            yield np.array(real_img), np.array(real_y), np.array(fake_y)

def sample_image(image_num, gen, save_path):
    global SAVED_IMAGES
    attr = []
    for i in range(image_num): attr.append(ToEmbed('aqua hair', 'green eyes'))
    attr  = Variable(torch.cuda.FloatTensor(attr))
    noise = Variable(torch.randn(image_num, NOISE_DIM, 1, 1)).cuda()
    noise = Variable(torch.cuda.FloatTensor(noise))
    image = gen(noise, attr)

    image_list = []
    for img in image:
        img = (img.data-img.data.min()) / (img.data.max()-img.data.min())
        image_list.append(img)

    haha = os.path.join(OUTDIR, 'images', 'aqua_hair_green_eyes_'+save_path+'.png')
    save_image(make_grid(image_list, nrow=5), haha)
    print ('{} images sampled, saving to {}'.format(SAMPLE_NUM, haha))
    SAVED_IMAGES.append(haha)
    if len(SAVED_IMAGES) > MAX_SAVE:
        os.remove(SAVED_IMAGES[0])
        SAVED_IMAGES = SAVED_IMAGES[1:]

G = Generator(len(ATTR))
D = Discriminator(len(ATTR))
G.initWeight()
D.initWeight()
G.cuda()
D.cuda()

# loss
BCE_loss = nn.BCELoss()

# optimizer
G_opt = optim.Adam(G.parameters(), lr=LEARNING_RATE)
D_opt = optim.RMSprop(D.parameters(), lr=LEARNING_RATE)

# records
history = {}
history['G_loss'] = []
history['D_loss'] = []

# loader
Loader = loader(DATA)

# train
real = torch.ones(BATCH_SIZE)
fake = torch.zeros(BATCH_SIZE)
real, fake = Variable(real.cuda()), Variable(fake.cuda())
steps_pre_epoch = len(DATA) // BATCH_SIZE
for epoch in range(EPOCH):
    for step in range(steps_pre_epoch):

        ###########
        # train G #
        ###########
        fake_list = []
        real_list = []
        correct   = []
        incorrect = []
        for i in range(2):
            image, y, fake_y = Loader.__next__()
            z        = random_noise()
            image    = Variable(torch.cuda.FloatTensor(image))
            y        = Variable(torch.cuda.FloatTensor(y))
            fake_y   = Variable(torch.cuda.FloatTensor(fake_y))
            z        = Variable(torch.cuda.FloatTensor(z))

            G_opt.zero_grad()

            fake_imgs = G(z, y)
            img_out, label_out = D(fake_imgs, y)

            img_loss   = BCE_loss(img_out.squeeze(), real)
            label_loss = BCE_loss(label_out.squeeze(), real)
            g_loss     = (img_loss + label_loss) / 2
            g_loss.backward()
            G_opt.step()

            fake_list.append(fake_imgs)
            real_list.append(image)
            correct.append(y)
            incorrect.append(fake_y)
        
        ###########
        # train D #
        ###########
        D_opt.zero_grad()

        # for real image, correct labels
        img_out_1, label_out_1 = D(real_list[0], correct[0])
        loss_1 = BCE_loss(img_out_1.squeeze(), real) + BCE_loss(label_out_1.squeeze(), real)

        # for real image, wrong labels
        img_out_2, label_out_2 = D(real_list[1], incorrect[1])
        loss_2 = BCE_loss(img_out_2.squeeze(), real) + BCE_loss(label_out_2.squeeze(), fake)

        # for fake image, correct labels
        img_out_3, label_out_3 = D(fake_list[0].detach(), correct[0])
        loss_3 = BCE_loss(img_out_3.squeeze(), fake) + BCE_loss(label_out_3.squeeze(), real)

        # for fake image, wrong labels
        img_out_4, label_out_4 = D(fake_list[1].detach(), incorrect[1])
        loss_4 = BCE_loss(img_out_4.squeeze(), fake) + BCE_loss(label_out_4.squeeze(), fake)

        d_loss = (loss_1 + loss_2 + loss_3 + loss_4) / 8

        d_loss.backward()
        D_opt.step()

        print ('[Epoch {:3}] [Step {:4}] [G_loss {:.6f}] [D_loss {:.6f}]'.format(epoch+1, step+1, g_loss.item(), d_loss.item()))

        if step%100 == 0:
            history['G_loss'].append(g_loss.item())
            history['D_loss'].append(d_loss.item())
            sample_image(SAMPLE_NUM, G, 'epoch-{:03}-step{:04}'.format(epoch+1, step+1))

        if epoch > 20 and epoch%INTERVAL == 0 and step == steps_pre_epoch-1:
            torch.save(G.state_dict(), os.path.join(OUTDIR, 'Generator-E{:03}.model'.format(epoch+1)))
            torch.save(D.state_dict(), os.path.join(OUTDIR, 'Discriminator-E{:03}.model'.format(epoch+1)))
            with open(os.path.join(OUTDIR, 'history.pkl'), 'wb') as f:
                pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
            print ('Model Saved!')
