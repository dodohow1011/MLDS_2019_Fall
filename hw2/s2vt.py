import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from DataSet import DataSet
import numpy as np

class Net(nn.module):
    def __init__():
        super(Net, self).__init__(batch_size, total_words, decoder_length)

        self.video_frame = 80
        self.batch_size = batch_size
        self.decoder_length = decoder_length
        self.total_words = total_words
        
        # model
        self.lstm1 = nn.LSTMCell(input_size=4096, hidden_size=256)
        nn.init.orthogonal(self.lstm1.weight_hh);
        nn.init.orthogonal(self.lstm1.weight_ih);
        self.lstm2 = nn.LSTMCell(input_size=512, hidden_size=256)
        nn.init.orthogonal(self.lstm2.weight_hh);
        nn.init.orthogonal(self.lstm2.weight_ih);
        
        self.one_hot = nn.Linear(256, total_words)

    def forward(self, feat, caption, caption_one_hot):
        # initialize
        h1 = Variable(torch.zeros(self.batch_size, self.lstm1.hidden_size)).cuda();
        h2 = Variable(torch.zeros(self.batch_size, self.lstm2.hidden_size)).cuda();
        c1 = Variable(torch.zeros(self.batch_size, self.lstm1.hidden_size)).cuda();
        c2 = Variable(torch.zeros(self.batch_size, self.lstm2.hidden_size)).cuda();

        # <pad>
        padding_lstm1 = Variable(torch.zeros(self. batch_size, 4096)).cuda();
        padding_lstm2 = Variable(torch.zeros(self.batch_size, 256)).cuda();

        total_loss=Variable(torch.FloatTensor([0])).cuda();
        
        # encoding
        for step in range(self.video_frame):
            h1, c1 = self.lstm1(feat[:,step,:], (h1, c1))
            h2, c2 = self.lstm2(torch.cat(padding_lstm2, h1), (h2, c2))
        
        # decoding
        for step in range(self.decoder_length):          
            h1, c1 = self.lstm1(padding_lstm1, (h1, c1))
            h2, c2 = self.lstm2(torch.cat(caption[:,step,:], h1), (h2, c2))

            word_loss = nn.CrossEntropyLoss(nn.Softmax(self.one_hot(h2), caption_one_hot)
            total_loss = total_loss + word_loss

        total_loss = total_loss / self.batch_size
        return total_loss

def train():
    batch_size = 4
    epochs = 100

    data = DataSet(word2vec='word_model/model.wv', training_features='MLDS_hw2_1_data/training_data/id.txt', training_label='MLDS_hw2_1_data/training_label.json', testing_features='MLDS_hw2_1_data/testing_data/id.txt', testing_label='MLDS_hw2_1_data/testing_label.json', batch_size=batch_size)
    
    total_words = data.VocabSize()
    decoder_length = data.MaxSeqLength()
    
    model = Net(batch_size=batch_size, total_words=total_words, decoder_length=decoder_length)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []

    for epoch in range(epochs):
        feature, caption = data.next_batch()
        feature, caption = Variable(torch.FloatTensor(feature)), Variable(torch.FloatTensor(caption))

        loss = model(feature, caption, caption_one_hot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.data[0])

        print ('===> epoch: {}, loss: {:.4f}'.format(epoch+1, loss))
        if epoch%10 == 0:
            torch.save(model.state_dict(), "./model_checkpoint/s2vt.pytorch.{0}".format(epoch))

    torch.save(model.state_dict(), "./model_checkpoint/s2vt.pytorch")

def main():
    train()


if __name__ == '__main__':
    main()

