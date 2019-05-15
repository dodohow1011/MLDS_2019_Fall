import torch
import pickle
from scipy import misc
import torchvision.transforms as transforms
from DCGAN import Generator, Discriminator
from argparse import ArgumentParser
import torch.optim as optim
from random import shuffle
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from torchsummary import summary
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# initialize parameters
input_channel_num = 100
output_channel_num = 128
d_drop = 0.0
g_drop = 0.0
generated_img_num = 25

def next_batch(train_img_list, num_step, batch_size):
    while True:
        print("Shuffling data...")
        shuffle(train_img_list)
        print("Done")
        for step in range(num_step):
            img_list = list()
            for i in range(batch_size):
                img = misc.imread(train_img_list[step*batch_size+i])
                img_list.append(img)
            yield img_list

def generate_images(gen_img, epoch, gen_file):
    image_list = []
    for i in range(len(gen_img)):
        image = gen_img[i]
        image = (image.data-image.data.min()) / (image.data.max() - image.data.min())
        image_list.append(image)
    save_image(make_grid(image_list, nrow=5), os.path.join(gen_file, 'epoch-{}.png'.format(epoch+1)))

def train(epochs, batch_size, num_step, train_img_list, save_dir, generated_img_file, loss_file):
    print ("Number of epochs: " + str(epochs))
    print ("Batch Size: " + str(batch_size))
    print ("Steps per epoch: " + str(num_step))
    print ("--------------------")

    # transform to [-1,1]
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
    
    # model
    discriminator = Discriminator(output_channel_num, d_drop)
    generator = Generator(input_channel_num, output_channel_num, g_drop)
    discriminator.cuda()
    summary(discriminator, (3, 64, 64))
    generator.cuda()
    summary(generator, (input_channel_num, 1, 1))

    # optimizer
    D_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0002)
    G_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # labels
    real_label = Variable(torch.rand((batch_size,))).cuda().squeeze()*0.1 + 0.9
    fake_label = Variable(torch.full((batch_size,), 0)).cuda().squeeze()
    
    # generate batch
    gen = next_batch(train_img_list, num_step, batch_size)

    D_loss = []
    G_loss = []
    
    for epoch in range(epochs):
        for step in range(num_step):
            image = gen.__next__()
            for i in range(batch_size):
                image[i] = transform(image[i]).tolist()

            image = Variable(torch.FloatTensor(image)).cuda()

            print ("===> Epoch: " + str(epoch+1) + " Step: " + str(step+1), end=" ")
            
            # train discriminator
            for i in range(1):
                noise = Variable(torch.randn(batch_size, input_channel_num, 1, 1)).cuda()
                fake_image = generator(noise)
                discriminator.zero_grad()
                real_loss = discriminator.loss(discriminator(image).squeeze(), real_label)
                real_loss.backward()
                D_optimizer.step()

                fake_loss = discriminator.loss(discriminator(fake_image.detach()).squeeze(), fake_label)
                fake_loss.backward()
                D_optimizer.step()
                total_loss = real_loss + fake_loss
                D_loss.append(total_loss.item())
                print ("Discriminator Loss: {:.4f}".format(total_loss.item()), end=" ")

            # train generator
            for i in range(1):
                noise = Variable(torch.randn(batch_size, input_channel_num, 1, 1)).cuda()
                fake_image = generator(noise)
                generator.zero_grad()
                loss = generator.loss(discriminator(fake_image).squeeze(), real_label)
                loss.backward()
                G_optimizer.step()
                G_loss.append(loss.item())
                print ("Generator Loss: {:.4f}".format(loss.item()))
            
        # save model
        if save_dir is not None:
            print ("Saving checkpoint...")
            torch.save({'Generator': generator.state_dict(), 'Discriminator': discriminator.state_dict()}, os.path.join(save_dir, 'model_dcgan.pt-{}'.format(epoch+1)))
            print ("Done")
            
        # generate images
        if generated_img_file is not None:
            noise = Variable(torch.randn(generated_img_num, input_channel_num, 1, 1)).cuda()
            gen_img = generator(noise)
            generate_images(gen_img, epoch, generated_img_file)
        
    if save_dir is not None:
            torch.save({'Generator': generator.state_dict(), 'Discriminator': discriminator.state_dict()}, os.path.join(save_dir, 'model_dcgan.pt'))
    
    print ("Saving loss history...")
    with open(os.path.join(loss_file, 'dcgan_d_loss.pickle'), 'wb') as f:
        pickle.dump(D_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(loss_file, 'dcgan_g_loss.pickle'), 'wb') as f:
        pickle.dump(G_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
    print ("Done")

def main(args):
    mode = args.mode
    batch_size = args.batch_size
    epochs = args.epoch
    train_image_file = args.train
    save_dir = args.model
    generated_img_file = args.generated
    loss_file = args.loss_file
    checkpoint = args.checkpoint

    train_img_list = []

    if mode == 'train':
        with open(train_image_file, 'r') as f:
            for line in f:
                line = line.strip()
                train_img_list.append(line)
    
        num_step = len(train_img_list) // batch_size

    if mode == 'test':
        generator = Generator(input_channel_num, output_channel_num, g_drop)
        generator.load_state_dict(state_dict=torch.load(checkpoint)['Generator'])
        generator.cuda()
        
        if generated_img_file is not None:
            noise = Variable(torch.randn(generated_img_num, input_channel_num, 1, 1)).cuda()
            gen_img = generator(noise)
            generate_images(gen_img, 0, generated_img_file)


    elif mode == 'train':
        print ("Number of training datas: " + str(len(train_img_list)))
        train(epochs, batch_size, num_step, train_img_list, save_dir, generated_img_file, loss_file)

def parse():
    parser = ArgumentParser()
    parser.add_argument('--mode', required=True, help='train or test')
    parser.add_argument('--epoch', '-e', required=None, type=int, help='epochs')
    parser.add_argument('--batch_size', '-b', required=None, type=int, help='batch size')
    parser.add_argument('--train', '-t', default=None, help='the file that contains sentences')
    parser.add_argument('--checkpoint', default=None, help='model to be tested')
    parser.add_argument('--test', default=None, help='testing data')
    parser.add_argument('--model', '-m', default=None, help='trained model directory')
    parser.add_argument('--generated', '-g', default=None, help='gererated image')
    parser.add_argument('--loss_file', '-l', default=None, help='loss history')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())

