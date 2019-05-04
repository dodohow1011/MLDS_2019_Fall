import re
import os
import time
import numpy as np
from random import shuffle, choice
from argparse import ArgumentParser
from gensim.models import KeyedVectors
import sys


class DataSet:
    def __init__(self, word2vec, training_text, training_label ,batch_size, testing=None):
        self.word2vec           = KeyedVectors.load(word2vec, mmap='r')
        self.embedding_dim      = self.word2vec['<bos>'].shape[0]
        self.vocab_size         = len(self.word2vec.vocab) + 1 # <pad>
        self.training_in_txt    = []
        self.training_out_txt   = []
        self.training_out_id    = []
        self.testing            = []
        self.batch_size         = batch_size
        self.max_length         = None
        self.batch_index        = 0
        self.test_index         = 0
        self.training_num       = None
        self.max_batch_index    = None
        self.max_test_index     = None
        self.sentence_limit     = 20 # default max length of a sentence

        if testing is not None:
            with open(testing, 'r') as f:
                for line in f:
                    if line[:5] != '<bos>':
                        data = '<bos> '+line.strip()+' <eos>'
                    else:
                        data = line.strip()
                    data = data.split()
                    if len(data) > self.sentence_limit: data = data[-self.sentence_limit:]
                    while len(data) < self.sentence_limit: data.append('<pad>')
                    self.testing.append(data)

        if training_text is not None:
            with open (training_text, 'r') as f:
                for line in f:
                    self.training_in_txt.append(line.strip())

        if training_label is not None:
            with open(training_label, 'r') as f:
                for line in f:
                    self.training_out_txt.append(line.strip())

        #create mappings
        self.word2id = { '<pad>':0, '<bos>':1, '<eos>':2 }
        self.id2word = { 0:'<pad>', 1:'<bos>', 2:'<eos>' }
        curID = 3
        for word in self.word2vec.vocab:
            if word not in self.word2id:
                self.word2id[word]  = curID
                self.id2word[curID] = word
                curID +=1
        
        self.max_test_index = len(self.testing) // self.batch_size - 1
        if testing is not None: return

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
        print ('batch size    -> {}'.format(self.batch_size))
        print ('batch num     -> {}'.format(self.max_batch_index+1))

    def filter(self):
        print ('filtering data...')
        start = time.time()
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
        print ('took {}s'.format(time.time()-start))

        # pad the sequence here
        start = time.time()
        print ('padding sequences and creating id sequences...')
        for i in range(len(self.training_in_txt)):
            while len(self.training_in_txt[i]) < self.sentence_limit:
                self.training_in_txt[i].append('<pad>')
            while len(self.training_out_txt[i]) < self.sentence_limit:
                self.training_out_txt[i].append('<pad>')
            tmp = []
            for word in self.training_out_txt[i]:
                tmp.append(self.Word2ID(word))
            self.training_out_id.append(np.array(tmp))
        self.training_out_id = np.array(self.training_out_id)
        print ('took {}s'.format(time.time()-start))

    def data2vec(self, sentences):
        sentences_vec = []
        for sent in sentences:
            text_vec = []
            for word in sent:
                word_vec = self.Word2Vec(word)
                text_vec.append(word_vec)
            sentences_vec.append(np.array(text_vec))
        return np.array(sentences_vec)

    def Word2ID(self, word):
        return self.word2id[word]

    def ID2Word(self, ID):
        return self.id2word[ID]

    def Word2Vec(self, word):
        if word == '<pad>': return np.zeros(self.embedding_dim)
        try:
            return self.word2vec[word]
        except KeyError:
            return np.zeros(self.embedding_dim)

    def VocabSize(self):
        return self.vocab_size

    def MaxSeqLength(self):
        return self.sentence_limit

    def EmbeddingDim(self):
        return self.embedding_dim

    def next_batch(self):
        if self.batch_index == 0:
            print ('shuffling data...')
            start = time.time()
            c = list(zip(self.training_in_txt, self.training_out_txt, self.training_out_id))
            shuffle(c)
            self.training_in_txt, self.training_out_txt, self.training_out_id = zip(*c)
            print ('took {}s'.format(time.time()-start))

        '''
        shape:
            sents     : (batch size, max seq length, latent dimension)
            responses : (batch size, max seq length, latent dimension)
            onehot    : (batch size, max seq length, vocab size)
        '''
        index = self.batch_index*self.batch_size
        sents = self.training_in_txt[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size]
        sents = self.data2vec(sents)
        sent_ids = self.training_out_id[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size]
        responses = self.training_out_txt[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size]  
        responses = self.data2vec(responses)

        self.batch_index += 1
        if self.batch_index > self.max_batch_index: self.batch_index = 0
        return sents, responses, sent_ids

    def test_batch(self):
        '''
        shape:
            sents     : (batch size, max seq length, latent dimension)
        '''
        index = self.test_index*self.batch_size
        sents = self.testing[self.test_index*self.batch_size:(self.test_index+1)*self.batch_size]
        sents = self.data2vec(sents)
        # sents = self.padSeqVec(sents)

        self.test_index += 1
        if self.test_index > self.max_test_index: return None
        return sents

    def padSeqVec(self, labels):
        print ('test', self.sentence_limit)
        for i in range(len(labels)):
            if len(labels[i]) > self.sentence_limit: labels[i] = labels[i][-self.sentence_limit:]
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
            for ID in sentence:
                OnehotWord = np.zeros(self.vocab_size)
                OnehotWord[ID] = 1
                OnehotSentence.append(OnehotWord)
            OnehotLabels.append(np.array(OnehotSentence))
        return np.array(OnehotLabels)

    def IDs2VECs(self, ids):
        vecs = []
        for ID in ids:
            vecs.append(self.Word2Vec(self.ID2Word(ID)))
        return np.array(vecs)

def main(args):
    training_text     = args.train
    training_response = args.label
    word2vec          = args.model
    
    dataset = DataSet(word2vec, training_text, training_response, 32)

    exp = 20
    start = time.time()
    for i in range(exp):
        s = time.time()
        sent, response, onehot = dataset.next_batch()
        print ('{}s elapsed'.format(time.time()-s))
    end = time.time()
    print (sent.shape)
    print (response.shape)
    print (onehot.shape)
    print ('ave time for computing batch: {}'.format((end-start)/exp))

def parse():
    parser = ArgumentParser()
    parser.add_argument('--train', '-t',  default='ProcessedData/train.txt',    help='the file that contains sentences')
    parser.add_argument('--label', '-l',  default='ProcessedData/label.txt',    help='the file that contains responses')
    parser.add_argument('--model', '-m',  default='WordModel/WordModel_256.wv', help='the pretrained word2vec model')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())
