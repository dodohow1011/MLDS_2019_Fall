import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from chatbot_model import Encoder, Decoder
from DataSet import DataSet
import numpy as np
from argparse import ArgumentParser
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(batch_size, data, epochs, save_dir):
    batch_size = batch_size
    seq_size = data.MaxSeqLength()
    hidden_size = data.EmbeddingDim()
    n_layers = 2
    dropout = 0

    encoder = Encoder(batch_size=batch_size, seq_size=seq_size, word_emb_size=hidden_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
    encoder.cuda()

    decoder = Decoder(batch_size=batch_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
    decoder.cuda()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)  
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

    training_data_size = data.training_num
    batch_per_epoch    = training_data_size // batch_size

    
    for epoch in range(epochs):
        for batch in range(batch_per_epoch):
            sents, response, sent_ids = data.next_batch()
            sents, response, sent_ids = Variable(torch.FloatTensor(sents)), Variable(torch.FloatTensor(response)), Variable(torch.LongTensor(sent_ids))
            sents = sents.view(20, 64, -1)
            response = response.view(20, 64, -1)
            if torch.cuda.is_available():
                sents, response, sent_ids = sents.cuda(), response.cuda(), sent_ids.cuda()
            
            # print (caption_one_hot.shape)
            encoder_output, encoder_h = encoder(sents)
            decoder_h = encoder_h
            total_loss = Variable(torch.FloatTensor([0])).cuda()

            loss = nn.CrossEntropyLoss(reduce=None)
            loss_list = []
            for word in range(seq_size-1):
                # teacher forcing
                decoder_input = response[word, :, :].view(1, batch_size, -1)

                decoder_output, decoder_h = decoder(decoder_input, decoder_h)
                decoder_output = decoder_output.view(batch_size, -1)

                # calculate loss
                word_loss = loss(decoder.one_hot(decoder_output), sent_ids[:,word+1])
                total_loss += torch.sum(word_loss) / batch_size
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            total_loss.backward()

            loss_list.append(total_loss.item())

            clip = 50.0
            _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

            encoder_optimizer.step()
            decoder_optimizer.step()


            print ('===> epoch: {}, steps: {}, loss: {:.4f}'.format(epoch+1, batch, total_loss.item()))
            
            if (batch+1)%5000 == 0 and save_dir is not None:
                print ("Saving checkpoint-{}".format(batch+1))
                torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, os.path.join(save_dir, 'model_2layer.pytorch-{}-{}'.format(epoch+1, batch+1)))

    if save_dir is not None:
        print ("Saving checkpoint")
        torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, os.path.join(save_dir, 'model_2layer.pytorch'))
        with open(os.path.join(save_dir, 'loss_2layer.history', 'w')) as f:
            pickle.dump(loss_list, f, protocol=pickle.HIGHEST_PROTOCOL)


def test(args):
    
    batch_size = args.batchsize
    word_model = args.wordmodel
    checkpoint = args.checkpoint
    test_txt = args.test


    data = DataSet(word2vec=word_model, training_text=None, training_label=None, batch_size=batch_size, testing=test_txt)

    hidden_size = data.EmbeddingDim()
    total_words = data.VocabSize()
    seq_size  = data.MaxSeqLength()
    n_layers = 2
    dropout = 0
    
    print ('loading {}'.format(checkpoint))
    encoder = Encoder(batch_size=batch_size, seq_size=seq_size, word_emb_size=hidden_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
    encoder.cuda()
    encoder.load_state_dict(state_dict=torch.load(checkpoint)['encoder'])

    decoder = Decoder(batch_size=batch_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
    decoder.cuda()
    decoder.load_state_dict(state_dict=torch.load(checkpoint)['decoder'])

    output_txt = open(args.output, 'w')
    test_batch = data.test_batch()
    k = 1
    while test_batch is not None:
        output = []
        print ('{} responses generated'.format(k))
        test_batch = Variable(torch.FloatTensor(test_batch))
        test_batch = test_batch.view(seq_size, 1, -1)
        test_batch = test_batch.cuda()
        encoder_output, encoder_h = encoder.inference(test_batch)
        
        pre_word = data.Word2Vec('<bos>')
        decoder_h = encoder_h
        for word in range(seq_size-1):

            pre_word = Variable(torch.FloatTensor(pre_word))
            pre_word = pre_word.cuda()
            pre_word = pre_word.reshape((1, 1, hidden_size))
            decoder_h, word_id = decoder.inference(pre_word, decoder_h)
            word = data.ID2Word(word_id.data.cpu().numpy()[0])
            pre_word = data.Word2Vec(word)
            output.append(word)
        sentence = ' '.join(output)
        # sentence = sentence.replace('<bos>', '')
        # sentence = sentence.replace('<eos>', '')
        # sentence = sentence.replace('<pad>', '')
        
        output_txt.write(sentence + '\n')
        test_batch = data.test_batch()
        k += 1

    output_txt.close()

def main(args):

    batch_size  = args.batchsize
    epochs      = args.epoch
    word_model  = args.wordmodel
    train_text  = args.train
    train_reply = args.label
    save_dir    = args.output
    
    data = DataSet(word2vec=word_model, training_text=train_text, training_label=train_reply, batch_size=batch_size)
    if args.mode == 'train':
        train(batch_size, data, epochs, save_dir)
    elif args.mode =='test':
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
