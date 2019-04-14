import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from DataSet import DataSet
import numpy as np
from argparse import ArgumentParser
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 4
epochs = 200

data = DataSet(word2vec='word_model/model.wv', training_features='MLDS_hw2_1_data/training_data/id.txt', training_label='MLDS_hw2_1_data/training_label.json', testing_features='MLDS_hw2_1_data/testing_data/id.txt', testing_label='MLDS_hw2_1_data/testing_label.json', batch_size=batch_size)

word2vec_size = data.EmbeddingDim()
hidden_size = word2vec_size

class Net(nn.Module):
    def __init__(self, batch_size, total_words, decoder_length):
        super(Net, self).__init__()

        self.video_frame = 80
        self.batch_size = batch_size
        self.decoder_length = decoder_length
        self.total_words = total_words
        
        # model
        self.lstm1 = nn.LSTMCell(input_size=4096, hidden_size=hidden_size)
        self.lstm2 = nn.LSTMCell(input_size=2*hidden_size, hidden_size=hidden_size)

        self.lstm_decode1 = nn.LSTMCell(input_size=4096, hidden_size=hidden_size)
        self.lstm_decode2 = nn.LSTMCell(input_size=hidden_size+word2vec_size, hidden_size=hidden_size)

        # nn.init.orthogonal(self.lstm1.weight_hh);
        # nn.init.orthogonal(self.lstm1.weight_ih);
        # nn.init.orthogonal(self.lstm2.weight_hh);
        # nn.init.orthogonal(self.lstm2.weight_ih);
        # nn.init.orthogonal(self.lstm_decode1.weight_hh);
        # nn.init.orthogonal(self.lstm_decode1.weight_ih);
        # nn.init.orthogonal(self.lstm_decode2.weight_hh);
        # nn.init.orthogonal(self.lstm_decode2.weight_ih);
        
        self.one_hot = nn.Linear(hidden_size, total_words)

    def forward(self, feat, caption, caption_one_hot, keep_rate):
        # initialize
        h1 = Variable(torch.zeros(self.batch_size, self.lstm1.hidden_size)).cuda()
        h2 = Variable(torch.zeros(self.batch_size, self.lstm2.hidden_size)).cuda()
        c1 = Variable(torch.zeros(self.batch_size, self.lstm1.hidden_size)).cuda()
        c2 = Variable(torch.zeros(self.batch_size, self.lstm2.hidden_size)).cuda()

        # <pad>
        padding_lstm1 = Variable(torch.zeros(self.batch_size, 4096)).cuda();
        padding_lstm2 = Variable(torch.zeros(self.batch_size, hidden_size)).cuda();

        total_loss=Variable(torch.FloatTensor([0])).cuda();
        loss = nn.CrossEntropyLoss(reduce=None)
        # encoding
        h2_list = []
        for step in range(self.video_frame):
            h1, c1 = self.lstm1(feat[:,step,:], (h1, c1))
            h2, c2 = self.lstm2(torch.cat((padding_lstm2, h1), 1), (h2, c2))
            h2_list.append(h2.tolist())

        # attention
        h2_list = Variable(torch.FloatTensor(h2_list)).cuda()
        h2_list = h2_list.transpose(0, 1)
        
        # decoding
        for step in range(self.decoder_length-1):          
            h1, c1 = self.lstm_decode1(padding_lstm1, (h1, c1))

            # schedule sampling
            coin = np.random.randint(100)
            if step > 0 and coin >= 100:
                word_ids = np.argmax(one_hot_word.data.cpu().numpy(), axis=1)
                word_vec = data.IDs2VECs(word_ids)
                word_vec = Variable(torch.FloatTensor(word_vec)).cuda()
                h2, c2 = self.lstm_decode2(torch.cat((word_vec, h1), 1), (h2, c2))
            else:
                h2, c2 = self.lstm_decode2(torch.cat((caption[:,step,:], h1), 1), (h2, c2))

            one_hot_word = self.one_hot(h2)
            word_loss    = loss(one_hot_word, torch.max(caption_one_hot[:,step+1,:], 1)[1])
            total_loss  += torch.sum(word_loss) / self.batch_size

            # attention
            h2 = h2.reshape((self.batch_size, -1, 1))
            attn = torch.bmm(h2_list, h2)
            attn = F.softmax(attn.reshape((self.batch_size, -1)), dim=1)
            h2 = torch.bmm(h2_list.transpose(1, 2), attn.reshape((self.batch_size, -1, 1))).reshape((self.batch_size, -1))

        return total_loss

    def inference(self, feat): 
        # initialize
        h1 = Variable(torch.zeros(1, self.lstm1.hidden_size)).cuda()
        h2 = Variable(torch.zeros(1, self.lstm2.hidden_size)).cuda()
        c1 = Variable(torch.zeros(1, self.lstm1.hidden_size)).cuda()
        c2 = Variable(torch.zeros(1, self.lstm2.hidden_size)).cuda()
        
        sentence = []

        # <pad>
        padding_lstm1 = Variable(torch.zeros(1, 4096)).cuda()
        padding_lstm2 = Variable(torch.zeros(1, hidden_size)).cuda()

        # encoding
        h2_list = []
        for step in range(self.video_frame):
            h1, c1 = self.lstm1(feat[:,step,:], (h1, c1))
            h2, c2 = self.lstm2(torch.cat((padding_lstm2, h1), 1), (h2, c2))
            h2_list.append(h2.tolist())

        # attention
        h2_list = Variable(torch.FloatTensor(h2_list))
        h2_list = h2_list.transpose(0, 1)
        
        # decoding
        previous_word = data.Word2Vec('<BOS>')
        
        for step in range(self.decoder_length-1):          
            previous_word = Variable(torch.FloatTensor(previous_word))
            previous_word = previous_word.reshape((1, word2vec_size))
            previous_word = previous_word.cuda()
            h1, c1 = self.lstm_decode1(padding_lstm1, (h1, c1))
            h2, c2 = self.lstm_decode2(torch.cat((previous_word, h1), 1), (h2, c2))
            
            word_one_hot = self.one_hot(h2)
            _, word_id = torch.max(word_one_hot, 1)
            word = data.ID2Word(word_id.data.cpu().numpy()[0])
            sentence.append(word)

            previous_word = data.Word2Vec(word)
            # attention
            h2 = h2.reshape((1, -1, 1))
            attn = torch.bmm(h2_list, h2)
            attn = F.softmax(attn.reshape((1, -1)), dim=1)
            h2 = torch.bmm(h2_list.transpose(1, 2), attn.reshape((1, -1, 1))).reshape((1, -1))

        return sentence

def train():
    
    total_words = data.VocabSize()
    decoder_length = data.MaxSeqLength()
    
    model = Net(batch_size=batch_size, total_words=total_words, decoder_length=decoder_length)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []

    training_data_size = data.training_num
    batch_per_epoch = training_data_size // batch_size
    for epoch in range(epochs):
        for batch in range(batch_per_epoch):
            feature, caption, caption_one_hot = data.next_batch()
            feature, caption, caption_one_hot = Variable(torch.FloatTensor(feature)), Variable(torch.FloatTensor(caption)), Variable(torch.LongTensor(caption_one_hot))
            feature, caption, caption_one_hot = feature.cuda(), caption.cuda(), caption_one_hot.cuda()
        
            # print (caption_one_hot.shape)
            loss = model(feature, caption, caption_one_hot, 99-99/epochs*epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            print ('===> epoch: {}, steps: {}, loss: {:.4f}, sampling rate: {:.4f}'.format(epoch+1, batch, loss.item(), 0))
        if epoch%10 == 0:
           torch.save(model.state_dict(), './model/s2vt_att.pytorch-{}'.format(epoch))

    torch.save(model.state_dict(), './model/s2vt_att.pytorch')

def test(checkpoint, out):

    total_words = data.VocabSize()
    decoder_length = data.MaxSeqLength()
    
    model = Net(batch_size=batch_size, total_words=total_words, decoder_length=decoder_length)
    model.cuda()

    model.load_state_dict(state_dict=torch.load(checkpoint))
    test_data, test_id = data.test_batch()
    # test_data, _, _ = data.next_batch()
    i = 0
    output_txt = open(out, "w")
    
    for test_feat in test_data:

        test_feat = Variable(torch.FloatTensor(test_feat))
        if (test_feat.shape[0] != 80):
            continue
        test_feat = test_feat.reshape((1, 80, -1))
        test_feat = test_feat
        output = model.inference(test_feat)
        sentence = ' '.join(output)
        sentence = sentence.replace('<BOS>', '')
        sentence = sentence.replace('<EOS>', '')
        sentence = sentence.replace('<PAD>', '')
        
        output_txt.write(test_id[i] + ',' + sentence + '\n')
        i += 1
        print (sentence)

def main(args):
    if args.mode == 'train':
        train()
    else:
        checkpoint = args.checkpoint
        output = args.output
        test(checkpoint, output)



def parse():
    parser = ArgumentParser()
    parser.add_argument('--mode', default='train', help='train or test')
    parser.add_argument('--checkpoint', help='model to be tested')
    parser.add_argument('--output', help='output file')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())

