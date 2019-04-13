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

data = DataSet(word2vec='word_model/model_32.wv', training_features='MLDS_hw2_1_data/training_data/id.txt', training_label='MLDS_hw2_1_data/training_label.json', testing_features='MLDS_hw2_1_data/testing_data/id.txt', testing_label='MLDS_hw2_1_data/testing_label.json', batch_size=batch_size)

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
            h1, c1 = self.lstm1(padding_lstm1, (h1, c1))

            # schedule sampling
            coin = np.random.randint(100)
            if step > 0 and coin >= 100:
                word_ids = np.argmax(one_hot_word.data.cpu().numpy(), axis=1)
                word_vec = data.IDs2VECs(word_ids)
                word_vec = Variable(torch.FloatTensor(word_vec)).cuda()
                h2, c2 = self.lstm2(torch.cat((word_vec, h1), 1), (h2, c2))
            else:
                h2, c2 = self.lstm2(torch.cat((caption[:,step,:], h1), 1), (h2, c2))

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
        padding_lstm1 = Variable(torch.zeros(1, 4096)).cuda();
        padding_lstm2 = Variable(torch.zeros(1, hidden_size)).cuda();

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
        previous_word = data.Word2Vec('<BOS>')
        p_candidate_list = []
        p_candidate_prob = []
        n_candidate_list = []
        n_candidate_prob = []
        answer_list = [[] for i in range(6)]
        answer_prob = [1 for i in range(6)]
        h_list = []
        c_list = []
        new_h_list = []
        new_c_list = []
        

        for step in range(self.decoder_length-1):
            if step == 0:          
                previous_word = Variable(torch.FloatTensor(previous_word))
                previous_word = previous_word.reshape((1, word2vec_size))
                previous_word = previous_word.cuda()
                h1, c1 = self.lstm1(padding_lstm1, (h1, c1))
                h2, c2 = self.lstm2(torch.cat((previous_word, h1), 1), (h2, c2))

                word_one_hot = self.one_hot(h2)
                word_one_hot = F.softmax(word_one_hot, dim=1)
                word_prob, word_id = torch.topk(word_one_hot, 3)
                tup = list(zip(word_prob[0].cpu().tolist(), word_id[0].cpu().tolist()))
                for _i, (_prob, _id) in enumerate(tup):
                    #new candidate multiple answer prob and wait to choose topk candidate
                    answer_list[_i].append(data.ID2Word(_id))
                    answer_prob[_i] *= _prob
                    p_candidate_list.append(data.ID2Word(_id))
                    p_candidate_prob.append(_prob)

                # attention
                h2 = h2.reshape((1, -1, 1))
                attn = torch.bmm(h2_list, h2)
                attn = F.softmax(attn.reshape((1, -1)), dim=1)
                h2 = torch.bmm(h2_list.transpose(1, 2), attn.reshape((1, -1, 1))).reshape((1, -1))

                for i in range(6):
                    h_list.append([h1,h2])
                    c_list.append([c1,c2])

                for i in range(6):
                    new_h_list.append([h1,h2])
                    new_c_list.append([c1,c2])
                

            else:
                n_candidate_list.clear()
                n_candidate_prob.clear()
                for i in range(len(p_candidate_list)):
                    previous_word = Variable(torch.FloatTensor(data.Word2Vec(p_candidate_list[i])))
                    previous_word = previous_word.reshape((1, word2vec_size))
                    previous_word = previous_word.cuda()

                    new_h_list[i][0], new_c_list[i][0] = self.lstm1(padding_lstm1, (h_list[i][0], c_list[i][0]))
                    new_h_list[i][1], new_c_list[i][1] = self.lstm2(torch.cat((previous_word, new_h_list[i][0]), 1), (h_list[i][1], c_list[i][1]))

                    word_one_hot = self.one_hot(new_h_list[i][1])
                    word_one_hot = F.softmax(word_one_hot, dim=1)
                    word_prob, word_id = torch.topk(word_one_hot, 3)
                    tup = list(zip(word_prob[0].cpu().tolist(), word_id[0].cpu().tolist()))

                    for _i, (_prob, _id) in enumerate(tup):
                        #new candidate multiple answer prob and wait to choose topk candidate
                        n_candidate_list.append(data.ID2Word(_id))
                        n_candidate_prob.append(_prob*answer_prob[int(i/3)])

                if any(_id==0 for _id in n_candidate_list):
                    break

                temp_answer_list = [[] for i in range(6)]
                temp_answer_prob = [1 for i in range(6)]
                if len(n_candidate_prob) > 6:
                    n_candidate_index = sorted(range(len(n_candidate_prob)), key=lambda j: n_candidate_prob[j],reverse=True)[:6]
                    p_candidate_list.clear()
                    p_candidate_prob.clear()
                    for _i, c_index in enumerate(n_candidate_index):
                        ans_index = int(c_index/3)
                        word = n_candidate_list[c_index]

                        temp = answer_list[ans_index].copy()
                        temp.append(word)
                        temp_answer_list[_i] = temp
                        temp_answer_prob[_i] = answer_prob[ans_index] * n_candidate_prob[c_index]
                        p_candidate_list.append(word)
                        p_candidate_prob.append(n_candidate_prob[c_index])

                        # attention
                        temp_h2 = h_list[ans_index][1]
                        temp_h2 = temp_h2.reshape((1, -1, 1))
                        attn = torch.bmm(h2_list, temp_h2)
                        attn = F.softmax(attn.reshape((1, -1)), dim=1)
                        new_h_list[_i][0] = h_list[ans_index][0]
                        new_h_list[_i][1] = torch.bmm(h2_list.transpose(1, 2), attn.reshape((1, -1, 1))).reshape((1, -1))
                        new_c_list[_i][0] = c_list[ans_index][0]
                        new_c_list[_i][1] = c_list[ans_index][1]

                answer_list = temp_answer_list.copy()
                answer_prob = temp_answer_prob.copy()
                h_list = new_h_list.copy()
                c_list = new_c_list.copy()
            '''
            for i in range(3):
                previous_word = Variable(torch.FloatTensor(p_candidate_list[i]))
                previous_word = previous_word.reshape((1, word2vec_size))
                previous_word = previous_word.cuda()
                h_list[i][0], c_list[i][0] = self.lstm_decode1(padding_lstm1, (h_list[i][0], c_list[i][0]))
                h_list[i][1], c_list[i][1] = self.lstm_decode2(torch.cat((previous_word, h_list[i][0]), 1), (h_list[i][1], c_list[i][1]))

                word_one_hot = self.one_hot(h_list[i][1])
                word_one_hot = F.softmax(word_one_hot, dim=1)
                word_prob, word_id = torch.max(word_one_hot, 1)
                word = data.ID2Word(word_id.data.cpu().numpy()[0])
                answer_list[i].append(word)
                answer_prob[i] = answer_prob[i]*word_prob

                p_candidate_list[i] = data.Word2Vec(word)
                # attention
                h_list[i][1] = h_list[i][1].reshape((1, -1, 1))
                attn = torch.bmm(h2_list, h_list[i][1])
                attn = F.softmax(attn.reshape((1, -1)), dim=1)
                h_list[i][1] = torch.bmm(h2_list.transpose(1, 2), attn.reshape((1, -1, 1))).reshape((1, -1))
            '''
        return answer_list[answer_prob.index(max(answer_prob))]

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
           torch.save(model.state_dict(), './model_checkpoint_embed_32/s2vt_att.pytorch-{}'.format(epoch))

    torch.save(model.state_dict(), './model_checkpoint_embed_32/s2vt_att.pytorch')

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
        test_feat = test_feat.cuda()
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
