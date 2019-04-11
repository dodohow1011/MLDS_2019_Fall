import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from DataSet import DataSet
import numpy as np
import sys
import os

batch_size = 4
epochs = 100
hidden_size = 256

data = DataSet(word2vec='word_model/model.wv', training_features='MLDS_hw2_1_data/training_data/id.txt', training_label='MLDS_hw2_1_data/training_label.json', testing_features='MLDS_hw2_1_data/testing_data/id.txt', testing_label='MLDS_hw2_1_data/testing_label.json', batch_size=batch_size)

word2vec_size = data.EmbeddingDim()
class Net(nn.Module):
    def __init__(self, batch_size, total_words, decoder_length):
        super(Net, self).__init__()

        self.video_frame = 80
        self.batch_size = batch_size
        self.decoder_length = decoder_length
        self.total_words = total_words
        
        # model
        self.lstm1 = nn.LSTMCell(input_size=4096, hidden_size=hidden_size)
        nn.init.orthogonal(self.lstm1.weight_hh);
        nn.init.orthogonal(self.lstm1.weight_ih);
        self.lstm2 = nn.LSTMCell(input_size=hidden_size+word2vec_size, hidden_size=hidden_size)

        nn.init.orthogonal(self.lstm2.weight_hh);
        nn.init.orthogonal(self.lstm2.weight_ih);
        
        self.one_hot = nn.Linear(hidden_size, total_words)

    def forward(self, feat, caption, caption_one_hot):
        # initialize
        h1 = Variable(torch.zeros(self.batch_size, self.lstm1.hidden_size)).cuda()
        h2 = Variable(torch.zeros(self.batch_size, self.lstm2.hidden_size)).cuda()
        c1 = Variable(torch.zeros(self.batch_size, self.lstm1.hidden_size)).cuda()
        c2 = Variable(torch.zeros(self.batch_size, self.lstm2.hidden_size)).cuda()

        # <pad>
        padding_lstm1 = Variable(torch.zeros(self.batch_size, 4096)).cuda();
        padding_lstm2 = Variable(torch.zeros(self.batch_size, hidden_size)).cuda();

        total_loss=Variable(torch.FloatTensor([0])).cuda();
        loss = nn.CrossEntropyLoss(reduce=False)
        # encoding
        for step in range(self.video_frame):
            h1, c1 = self.lstm1(feat[:,step,:], (h1, c1))
            h2, c2 = self.lstm2(torch.cat((padding_lstm2, h1), 1), (h2, c2))
        
        # decoding
        for step in range(self.decoder_length-1):          
            h1, c1 = self.lstm1(padding_lstm1, (h1, c1))
            h2, c2 = self.lstm2(torch.cat((caption[:,step,:], h1), 1), (h2, c2))
            word_loss = loss(self.one_hot(h2), torch.max(caption_one_hot[:,step+1,:], 1)[1])
            total_loss += torch.sum(word_loss) / self.batch_size

        return total_loss

    def inference(self, feat): 
        # initialize
        h1 = Variable(torch.zeros(1, self.lstm1.hidden_size)).cuda()
        h2 = Variable(torch.zeros(1, self.lstm2.hidden_size)).cuda()
        c1 = Variable(torch.zeros(1, self.lstm1.hidden_size)).cuda()
        c2 = Variable(torch.zeros(1, self.lstm2.hidden_size)).cuda()
        
        sentence = []

        # <pad>
        padding_lstm1 = Variable(torch.zeros(1, 4096)).cuda();
        padding_lstm2 = Variable(torch.zeros(1, hidden_size)).cuda();

        # encoding
        for step in range(self.video_frame):
            h1, c1 = self.lstm1(feat[:,step,:], (h1, c1))
            h2, c2 = self.lstm2(torch.cat((padding_lstm2, h1), 1), (h2, c2))
        
        # decoding
        previous_word = data.Word2Vec('<BOS>')
        
        for step in range(self.decoder_length-1):          
            previous_word = Variable(torch.FloatTensor(previous_word))
            previous_word = previous_word.reshape((1, word2vec_size))
            previous_word = previous_word.cuda()
            h1, c1 = self.lstm1(padding_lstm1, (h1, c1))
            h2, c2 = self.lstm2(torch.cat((previous_word, h1), 1), (h2, c2))
            
            word_one_hot = self.one_hot(h2)
            _, word_id = torch.max(word_one_hot, 1)
            word = data.ID2Word(word_id.data.cpu().numpy()[0])
            previous_word = data.Word2Vec(word)
            if step == 0:
                word = word.upper()
            sentence.append(word)


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
            loss = model(feature, caption, caption_one_hot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            print ('===> epoch: {}, steps: {}, loss: {:.4f}'.format(epoch+1, batch, loss.item()))
        if epoch%10 == 0:
           torch.save(model.state_dict(), './model_checkpoint/s2vt.pytorch-{}'.format(epoch))

    torch.save(model.state_dict(), './model_checkpoint/s2vt.pytorch')

def test(checkpoint):

    total_words = data.VocabSize()
    decoder_length = data.MaxSeqLength()
    
    model = Net(batch_size=batch_size, total_words=total_words, decoder_length=decoder_length)
    model.cuda()

    model.load_state_dict(state_dict=torch.load(checkpoint))
    test_data, test_id = data.test_batch()
    i = 0
    output_txt = open("./caption.txt", "w")
    
    for test_feat in test_data:

        test_feat = Variable(torch.FloatTensor(test_feat))
        if (test_feat.shape[0] != 80):
            continue
        test_feat = test_feat.reshape((1, 80, -1))
        test_feat = test_feat.cuda()
        output = model.inference(test_feat)
        sentence = ' '.join(output)
        sentence = sentence.replace('<BOS>', '')
        sentence = sentence.replace('<EOS>', '')
        sentence = sentence.replace('<PAD>', '')
        
        output_txt.write(test_id[i] + ',' + sentence + '\n')
        i += 1
        print (sentence)

def main():
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'test':
            if len(sys.argv) == 2:
                test("./model_checkpoint/s2vt.pytorch")
            else:
                checkpoint = sys.argv[2]
                test(checkpoint)
        else:
            print ("usage: type 'python3 s2vt.py train' to train the model and 'python3 s2vt.py test' to test the model")
    else:
        print ("usage: type 'python3 s2vt.py train' to train the model and 'python3 s2vt.py test' to test the model")

if __name__ == '__main__':
    # os.makedirs("model_checkpoint", exist_ok=True)
    main()

