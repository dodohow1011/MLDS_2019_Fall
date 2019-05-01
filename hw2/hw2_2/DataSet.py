import re
import os
import time
import numpy as np
from random import shuffle, choice
from argparse import ArgumentParser
from gensim.models import KeyedVectors
import sys


class DataSet:
    def __init__(self, word2vec, training_text, training_label ,batch_size=32):
        self.word2vec           = KeyedVectors.load(word2vec, mmap='r')
        self.embedding_dim      = self.word2vec['<bos>'].shape[0]
        self.vocab_size         = len(self.word2vec.vocab)
        self.training_in_txt    = []
        self.training_out_txt   = []
        self.batch_size         = batch_size
        self.max_length         = None
        self.batch_index        = 0
        self.training_num       = None
        self.max_batch_index    = None
        self.sentence_limit     = 20 # default max length of a sentence

        with open (training_text, 'r') as f:
            for line in f:
                self.training_in_txt.append(line.strip())

        with open(training_label, 'r') as f:
            for line in f:
                self.training_out_txt.append(line.strip())

        #create mappings
        self.word2id = {'<pad>':0, '<bos>':1, '<eos>':2}
        self.id2word = {0:'<pad>', 1:'<bos>', 2:'<eos>'}
        curID = 3
        for word in self.word2vec.vocab:
            if word not in self.word2id:
                self.word2id[word]  = curID
                self.id2word[curID] = word
                curID +=1
        
        '''
        filter out data with unknown words and those that exceeds length limit
        '''
        self.filter()
        self.training_num    = len(self.training_in_txt)
        self.max_batch_index = self.training_num // self.batch_size - 1

        print ('word model    -> {}'.format(word2vec))
        print ('embedding dim -> {}'.format(self.embedding_dim))
        print ('vocab size    -> {}'.format(self.vocab_size))
        print ('max length    -> {}'.format(self.sentence_limit))
        print ('training data -> {}'.format(self.training_num))
        print ('batch num     -> {}'.format(self.max_batch_index+1))

    def filter(self):
        print ('filtering data...', end='')
        temp_in    = []
        temp_out   = []
        max_length = 0
        for text_in, text_out in zip(self.training_in_txt, self.training_out_txt):
            splited_in  = text_in.split()
            splited_out = text_out.split()
            if len(splited_in)  > self.sentence_limit: continue
            if len(splited_out) > self.sentence_limit: continue

            flag = True
            for word in splited_in:
                try:
                    tmp = self.Word2ID(word)
                except KeyError:
                    # print ('{} oov'.format(word))
                    flag = False
                    break
            for word in splited_out:
                try:
                    tmp = self.Word2ID(word)
                except KeyError:
                    # print ('{} oov'.format(word))
                    flag = False
                    break

            if flag:
                temp_in.append(splited_in)
                temp_out.append(splited_out)
                max_length = max(max_length, len(splited_in)) 
                max_length = max(max_length, len(splited_out)) 

        self.training_in_txt  = temp_in
        self.training_out_txt = temp_out
        self.sentence_limit   = min(self.sentence_limit, max_length)

        print ('done')

    def data2vec(self, sentences):
        sentences_vec = []
        for sent in sentences:
            text_vec = []
            for word in sent:
                word_vec = self.Word2Vec(word)
                text_vec.append(word_vec)
            sentences_vec.append(text_vec)
        return sentences_vec

    def Word2ID(self, word):
        return self.word2id[word]

    def ID2Word(self, id):
        return self.id2word[id]

    def Word2Vec(self, word):
        if word == '<pad>': return np.zeros(self.embedding_dim)
        return self.word2vec[word]

    def VocabSize(self):
        return self.vocab_size

    def MaxSeqLength(self):
        return self.sentence_limit

    def EmbeddingDim(self):
        return self.embedding_dim

    def next_batch(self):
        if self.batch_index == 0:
            print ('shuffling data...')
            c = list(zip(self.training_in_txt, self.training_out_txt))
            shuffle(c)
            self.training_in_txt, self.training_out_txt = zip(*c)

        '''
        shape:
            sents     : (batch size, max seq length, latent dimension)
            responses : (batch size, max seq length, latent dimension)
            onehot    : (batch size, max seq length, vocab size)
        '''
        index = self.batch_index*self.batch_size
        sents = self.training_in_txt[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size]
        sents = self.data2vec(sents)
        sents = self.padSeqVec(sents)

        responses = self.training_out_txt[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size]  
        onehot    = self.toOnehot(responses)
        onehot    = self.padOneHot(onehot)
        responses = self.data2vec(responses)
        responses = self.padSeqVec(responses)

        self.batch_index += 1
        if self.batch_index > self.max_batch_index: self.batch_index = 0

        return sents, responses, onehot

    def padSeqVec(self, labels):
        for i in range(len(labels)):
            while len(labels[i]) < self.sentence_limit:
                labels[i].append(self.Word2Vec('<pad>'))
            labels[i] = np.array(labels[i])
        return np.array(labels)

    def padOneHot(self, labels):
        OneHotPad = np.zeros(self.vocab_size)
        OneHotPad[self.Word2ID('<pad>')] = 1
        for i in range(len(labels)):
            while len(labels[i]) < self.sentence_limit:
                labels[i].append(OneHotPad)
            labels[i] = np.array(labels[i])
        return np.array(labels)

    def toOnehot(self, texts):
        OnehotLabels = []
        for sentence in texts:
            OnehotSentence = []
            for word in sentence:
                OnehotWord = np.zeros(self.vocab_size)
                OnehotWord[self.Word2ID(word)] = 1
                OnehotSentence.append(OnehotWord)
            OnehotLabels.append(OnehotSentence)
        return OnehotLabels

    def IDs2VECs(self, ids):
        vecs = []
        for ID in ids:
            vecs.append(self.Word2Vec(self.ID2Word(ID)))
        return np.array(vecs)

def main(args):
    training_text     = args.train
    training_response = args.label
    word2vec          = args.model
    
    dataset = DataSet(word2vec, training_text, training_response)

    sent, response, onehot = dataset.next_batch()
    sent, response, onehot = dataset.next_batch()
    sent, response, onehot = dataset.next_batch()

    print (sent.shape)
    print (response.shape)
    print (onehot.shape)

def parse():
    parser = ArgumentParser()
    parser.add_argument('--train', '-t',  default='ProcessedData/train.txt',    help='the file that contains sentences')
    parser.add_argument('--label', '-l',  default='ProcessedData/label.txt',    help='the file that contains responses')
    parser.add_argument('--model', '-m',  default='WordModel/WordModel_256.wv', help='the pretrained word2vec model')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())
