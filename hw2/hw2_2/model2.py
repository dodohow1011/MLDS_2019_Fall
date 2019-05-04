import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable
from torchvision import transforms
from DataSet import DataSet
import numpy as np
import sys
import os
import pickle
from argparse import ArgumentParser

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Net(nn.Module):
    def __init__ (self, batch_size, total_words, max_length, hidden_size, data):
        super(Net,self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.total_words = total_words
        self.max_length = max_length
        self.lstm1 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.lstm2 = nn.LSTMCell(input_size=2*hidden_size, hidden_size=hidden_size)
        self.lstm_decode1 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.lstm_decode2 = nn.LSTMCell(input_size=2*hidden_size, hidden_size=hidden_size)
        self.data = data

        self.onehot = nn.Linear(hidden_size, total_words)

    def forward(self, feat, caption, ids):
        h1 = Variable(torch.zeros(self.batch_size, self.lstm1.hidden_size)).cuda()
        h2 = Variable(torch.zeros(self.batch_size, self.lstm2.hidden_size)).cuda()
        c1 = Variable(torch.zeros(self.batch_size, self.lstm1.hidden_size)).cuda()
        c2 = Variable(torch.zeros(self.batch_size, self.lstm2.hidden_size)).cuda()
        padding = Variable(torch.zeros(self.batch_size,self.hidden_size)).cuda()
        total_loss=Variable(torch.FloatTensor([0])).cuda()
            
        loss = nn.CrossEntropyLoss(reduce=None)

        # encode
        h2_list = []
        for step in range(self.max_length):
            h1, c1 = self.lstm1(feat[:,step,:],(h1,c1))
            h2, c2 = self.lstm2(torch.cat((padding, h1), 1), (h2, c2))
            h2_list.append(h2.tolist())

        for step in range(self.max_length-1):
            h1,c1 = self.lstm_decode1(padding, (h1,c1))
            h2,c2 = self.lstm_decode2(torch.cat((caption[:,step,:], h1), 1), (h2,c2))
            # word_loss = loss(self.onehot(h1),torch.max(caption_onehot[:,step+1,:],1)[1])
            word_loss = loss(self.onehot(h2),ids[:,step+1])
            total_loss += torch.sum(word_loss) / self.batch_size

        return total_loss

    def inference(self, feat): 
        if self.batch_size != 1:
            print ('set batch size to 1 when testing!')
            return
        # initialize
        h1 = Variable(torch.zeros(1, self.lstm1.hidden_size)).cuda()
        h2 = Variable(torch.zeros(1, self.lstm2.hidden_size)).cuda()
        c1 = Variable(torch.zeros(1, self.lstm1.hidden_size)).cuda()
        c2 = Variable(torch.zeros(1, self.lstm2.hidden_size)).cuda()
        
        sentence = []

        # <pad>
        padding = Variable(torch.zeros(1, self.hidden_size)).cuda();

        # encoding
        h2_list = []
        for step in range(self.max_length):
            h1, c1 = self.lstm1(feat[:,step,:], (h1, c1))
            h2, c2 = self.lstm2(torch.cat((padding, h1), 1), (h2, c2))
            h2_list.append(h2.tolist())

        # decoding
        previous_word = self.data.Word2Vec('<bos>')
        
        for step in range(self.max_length-1):          
            previous_word = Variable(torch.FloatTensor(previous_word))
            previous_word = previous_word.reshape((1, self.hidden_size))
            previous_word = previous_word.cuda()
            h1, c1 = self.lstm_decode1(padding, (h1, c1))
            h2, c2 = self.lstm_decode2(torch.cat((previous_word, h1), 1), (h2, c2))
            
            word_one_hot = self.onehot(h2)
            _, word_id = torch.max(word_one_hot, 1)
            word = self.data.ID2Word(word_id.data.cpu().numpy()[0])
            previous_word = self.data.Word2Vec(word)
            sentence.append(word)

        return sentence

def train(args):
    batch_size  = args.batchsize
    epochs      = args.epoch
    word_model  = args.wordmodel
    train_text  = args.train
    train_reply = args.label
    save_dir    = args.output

    if save_dir is not None:
        if not os.path.isdir(save_dir): os.mkdir(save_dir)

    data        = DataSet(word2vec=word_model, training_text=train_text, training_label=train_reply, batch_size=batch_size)

    hidden_size = data.EmbeddingDim()
    total_words = data.VocabSize()
    max_length  = data.MaxSeqLength()
    
    model       = Net(batch_size=batch_size, total_words=total_words, max_length=max_length, hidden_size=hidden_size, data=data)
    if torch.cuda.is_available(): model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []

    training_data_size = data.training_num
    batch_per_epoch    = training_data_size // batch_size
    for epoch in range(epochs):
        for batch in range(batch_per_epoch):
            feature, caption, caption_one_hot = data.next_batch()
            feature, caption, caption_one_hot = Variable(torch.FloatTensor(feature)), Variable(torch.FloatTensor(caption)), Variable(torch.LongTensor(caption_one_hot))
            if torch.cuda.is_available():
                feature, caption, caption_one_hot = feature.cuda(), caption.cuda(), caption_one_hot.cuda()
        
            # print (caption_one_hot.shape)
            loss = model(feature, caption, caption_one_hot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            print ('===> epoch: {}, steps: {}, loss: {:.4f}'.format(epoch+1, batch, loss.item()))
            if (batch+1)%5000 == 0 and save_dir is not None:
                torch.save(model.state_dict(), os.path.join(save_dir, 'model.pytorch-{}-{}'.format(epoch+1, batch+1)))

    if save_dir is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pytorch'))
        with open(os.path.join(save_dir, 'loss.history'), 'w') as f:
            pickle.dump(loss_list, f, protocol=pickle.HIGHEST_PROTOCOL)

def test(args):
    batch_size  = args.batchsize
    word_model  = args.wordmodel
    checkpoint  = args.checkpoint
    test_txt    = args.test

    data        = DataSet(word2vec=word_model, training_text=None, training_label=None, batch_size=batch_size, testing=test_txt)

    hidden_size = data.EmbeddingDim()
    total_words = data.VocabSize()
    max_length  = data.MaxSeqLength()
    
    model       = Net(batch_size=batch_size, total_words=total_words, max_length=max_length, hidden_size=hidden_size, data=data)
    if torch.cuda.is_available(): model.cuda()

    print ('loading {}'.format(checkpoint))
    model.load_state_dict(state_dict=torch.load(checkpoint))
    output_txt = open(args.output, 'w')
    test_batch = data.test_batch()
    k = 1
    while test_batch is not None:
        print ('{} responses generated'.format(k))
        test_batch = Variable(torch.FloatTensor(test_batch))
        test_batch = test_batch.cuda()
        output     = model.inference(test_batch)
        sentence = ' '.join(output)
        sentence = sentence.replace('<bos>', '')
        sentence = sentence.replace('<eos>', '')
        sentence = sentence.replace('<pad>', '')
        print (sentence)
        
        output_txt.write(sentence + '\n')
        test_batch = data.test_batch()
        k += 1

    output_txt.close()


def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)

def parse():
    parser = ArgumentParser()
    parser.add_argument('--mode', required=True, help='train or test')
    parser.add_argument('--epoch', '-e', required=None, type=int, help='epochs')
    parser.add_argument('--batchsize', '-b', required=True, type=int, help='batch size')
    parser.add_argument('--train', '-t', default=None, help='the file that contains sentences')
    parser.add_argument('--label', '-l', default=None, help='the file that contains responses')
    parser.add_argument('--wordmodel', '-wm', required=True, help='the pretrained word2vec model')
    parser.add_argument('--checkpoint', default=None, help='model to be tested')
    parser.add_argument('--test', default=None, help='testing data')
    parser.add_argument('--output', default=None, help='output directory')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())
