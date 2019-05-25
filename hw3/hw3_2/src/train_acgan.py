from ACGAN import Generator, Discriminator, NOISE_DIM
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
HAIR          = {}
EYES          = {}
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
    hairID = 0
    eyesID = 0
    for line in f:
        tmp = tuple(line.strip().split(','))
        DATA.append(tmp)
        if tmp[1] not in ATTR:
            ATTR[tmp[1]] = currentID
            currentID += 1
        if tmp[1] not in HAIR:
            HAIR[tmp[1]] = hairID
            hairID += 1
        if tmp[2] not in ATTR:
            ATTR[tmp[2]] = currentID
            currentID += 1
        if tmp[2] not in EYES:
            EYES[tmp[2]] = eyesID
            eyesID += 1

# save attribute dict
with open(os.path.join(OUTDIR, 'ATTR.pkl'), 'wb') as f:
    pickle.dump(ATTR, f, protocol=pickle.HIGHEST_PROTOCOL)

HairAttrNum = 0
EyesAttrNum = 0
for attr in list(ATTR):
    if attr.endswith('hair'): HairAttrNum += 1
    else: EyesAttrNum += 1

def ToEmbed(attr1, attr2):
    vec = np.zeros(len(ATTR))
    vec[ATTR[attr1]] = 1
    vec[ATTR[attr2]] = 1
    return vec.reshape((-1, 1, 1))

def HairEmbed(attr):
    vec = np.zeros(HairAttrNum)
    vec[HAIR[attr]] = 1
    return vec

def EyesEmbed(attr):
    vec = np.zeros(EyesAttrNum)
    vec[EYES[attr]] = 1
    return vec

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
            hair     = list()
            eyes     = list()
            for i in range(BATCH_SIZE):
                pth   = data[batch*BATCH_SIZE+i][0]
                attr1 = data[batch*BATCH_SIZE+i][1]
                attr2 = data[batch*BATCH_SIZE+i][2]
                real_img.append(transform(np.array(imread(pth))).tolist())
                real_y.append(ToEmbed(attr1, attr2))
                hair.append(HAIR[attr1])
                eyes.append(EYES[attr2])
            yield np.array(real_img), np.array(real_y), np.array(hair), np.array(eyes)

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
D = Discriminator(HairAttrNum, EyesAttrNum)
G.initWeight()
D.initWeight()
G.cuda()
D.cuda()

# loss
BCE_loss = nn.BCELoss()
auxilary_loss = nn.CrossEntropyLoss()

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

        # real     = Variable(torch.rand((BATCH_SIZE,))).cuda().squeeze()*0.1 + 0.9

        ###########
        # train G #
        ###########
        G_opt.zero_grad()

        image, y, hair, eyes = Loader.__next__()
        z        = random_noise()
        image    = Variable(torch.cuda.FloatTensor(image))
        y        = Variable(torch.cuda.FloatTensor(y))
        hair     = Variable(torch.cuda.LongTensor(hair))
        eyes     = Variable(torch.cuda.LongTensor(eyes))
        z        = Variable(torch.cuda.FloatTensor(z))

        gen_hair = Variable(torch.LongTensor(np.random.randint(0, HairAttrNum, (BATCH_SIZE,1))))
        one_hot_hair = torch.zeros(BATCH_SIZE, HairAttrNum).scatter_(1, gen_hair, 1).cuda()

        gen_eyes = Variable(torch.LongTensor(np.random.randint(0, EyesAttrNum, (BATCH_SIZE,1))))
        one_hot_eyes = torch.zeros(BATCH_SIZE, EyesAttrNum).scatter_(1, gen_eyes, 1).cuda()

        gen_label = torch.cat([one_hot_hair.unsqueeze(2).unsqueeze(3), one_hot_eyes.unsqueeze(2).unsqueeze(3)], 1)

        gen_hair = gen_hair.type(torch.LongTensor).cuda()
        gen_eyes = gen_eyes.type(torch.LongTensor).cuda()


        fake_imgs = G(z, gen_label)
        img_out, hair_out, eyes_out = D(fake_imgs)

        img_loss  = BCE_loss(img_out.squeeze(), real)
        hair_loss = auxilary_loss(hair_out, gen_hair.squeeze())
        eyes_loss = auxilary_loss(eyes_out, gen_eyes.squeeze())
        g_loss    = (img_loss + hair_loss + eyes_loss) / 3
        g_loss.backward()
        G_opt.step()

        ###########
        # train D #
        ###########
        D_opt.zero_grad()

        # for real images
        img_out_1, hair_out_1, eyes_out_1 = D(image)
        img_loss_1 = BCE_loss(img_out_1.squeeze(), real)
        hair_loss_1 = auxilary_loss(hair_out_1, hair)
        eyes_loss_1 = auxilary_loss(eyes_out_1, eyes)
        loss_1 = (img_loss_1 + hair_loss_1 + eyes_loss_1) / 3

        # for fake images
        img_out_2, hair_out_2, eyes_out_2 = D(fake_imgs.detach())
        img_loss_2 = BCE_loss(img_out_2.squeeze(), fake)
        hair_loss_2 = auxilary_loss(hair_out_2, gen_hair.squeeze())
        eyes_loss_2 = auxilary_loss(eyes_out_2, gen_eyes.squeeze())
        loss_2 = (img_loss_2 + hair_loss_2 + eyes_loss_2) / 3

        d_loss = (loss_1 + loss_2) / 2

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
