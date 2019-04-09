import json
import re
import os
import numpy as np
from random import shuffle, choice
from argparse import ArgumentParser
from gensim.models import KeyedVectors

class DataSet:
    def __init__(self, word2vec, training_features, training_label, testing_features, testing_label, batch_size=4):
        self.word2vec           = KeyedVectors.load(word2vec, mmap='r')
        self.vocab_size         = len(self.word2vec.vocab)
        self.training_label_txt = {} # training label in words with video id as key
        self.training_label_vec = {} # training label in vector form with video id as key
        self.testing_label_txt  = {}
        self.training_ids       = [] # the ids of the videos for training
        self.testing_ids        = [] # the ids of the videos for testing
        self.batch_size         = batch_size
        self.max_length         = None
        self.batch_index        = 0
        self.training_num       = None
        self.max_batch_index    = None
        self.sentence_limit     = 20 # default max length of a sentence

        with open(training_label, 'r') as f:
            data = json.loads(f.read())
        for movie in data:
            self.training_label_txt[movie['id']]  = []
            self.training_label_txt[movie['id']] += [('<BOS> '+re.sub(',', '', x[:-1].lower())+' <EOS>').strip().split() for x in movie['caption']]

        with open(testing_label, 'r') as f:
            data = json.loads(f.read())
        for movie in data:
            self.testing_label_txt[movie['id']]  = []
            self.testing_label_txt[movie['id']] += [('<BOS> '+re.sub(',', '', x[:-1].lower())+' <EOS>').strip().split() for x in movie['caption']]

        with open(training_features, 'r') as f:
            for line in f:
                self.training_ids.append(line.strip())

        with open(testing_features, 'r') as f:
            for line in f:
                self.testing_ids.append(line.strip())

        self.word2id = {'<PAD>':0, '<BOS>':1, '<EOS>':2}
        self.id2word = {0:'<PAD>', 1:'<BOS>', 2:'<EOS>'}
        
        curID = 3
        for word in self.word2vec.vocab:
            if word not in self.word2id:
                self.word2id[word]  = curID
                self.id2word[curID] = word
                curID += 1

        self.check_labels()

        self.training_num    = len(self.training_label_txt)
        self.max_batch_index = self.training_num // self.batch_size - 1

        print ('word model    -> {}'.format(word2vec))
        print ('vocab size    -> {}'.format(self.vocab_size))
        print ('max length    -> {}'.format(self.max_length))
        print ('training data -> {}'.format(self.training_num))
        print ('batch num     -> {}'.format(self.max_batch_index+1))

    def check_labels(self):
        self.max_length = 0
        for id, labels in self.training_label_txt.items():
            self.training_label_vec[id] = []
            ignore_sent = []
            for sentence in labels:
                sent_vec   = []
                legal_sent = True
                if len(sentence) > self.sentence_limit: legal_sent = False
                for word in sentence:
                    if not legal_sent:
                        ignore_sent.append(sentence)
                        break
                    try:
                        word_vec = self.word2vec[word]
                    except:
                        ignore_sent.append(sentence)
                        legal_sent = False
                        break
                    sent_vec.append(word_vec)
                if legal_sent:
                    self.training_label_vec[id].append(sent_vec)
                    self.max_length = max(self.max_length, len(sentence))
            for sent in ignore_sent:
                self.training_label_txt[id].remove(sent)

        '''
        for id in self.training_ids:
            print (len(self.training_label_vec[id][0]), end='') 
            print (' ', len(self.training_label_txt[id][0]))
        '''

    def Word2ID(self, word):
        return self.word2id[word]

    def ID2Word(self, id):
        return self.id2word[id]

    def Word2Vec(self, word):
        if word == '<PAD>': return np.zeros(256)
        return self.word2vec[word]

    def VocabSize(self):
        return self.vocab_size

    def MaxSeqLength(self):
        return self.max_length

    def next_batch(self):
        if self.batch_index == 0:
            shuffle(self.training_ids)

        '''
        shape:
            frames : (batch size, number of frames, number of features)
            labels : (batch size, max seq length,   latent dimension)
            onehot : (batch size, max seq length,   vocab size)
        '''

        ids    = self.training_ids[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size]

        videos = [ os.path.join('MLDS_hw2_1_data/training_data/feat', x+'.npy') for x in ids ]
        frames = np.array([ np.load(v) for v in videos ])

        index  = [ choice(np.arange(len(self.training_label_txt[ID]))) for ID in ids ]
        labels = [ self.training_label_vec[l][index[idx]] for idx, l in enumerate(ids) ]
        labels = self.padSeqVec(labels)

        texts  = [ self.training_label_txt[l][index[idx]] for idx, l in enumerate(ids) ]
        onehot = self.toOnehot(texts)
        onehot = self.padSeqOnehot(onehot)

        self.batch_index += 1
        if self.batch_index > self.max_batch_index: self.batch_index = 0

        return frames, labels, onehot

    def padSeqVec(self, labels):
        for i in range(len(labels)):
            while len(labels[i]) < self.max_length:
                labels[i].append(self.Word2Vec('<PAD>'))
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

    def padSeqOnehot(self, onehot):
        for i in range(len(onehot)):
            while len(onehot[i]) < self.max_length:
                OnehotPad = np.zeros(self.vocab_size)
                OnehotPad[self.Word2ID('<PAD>')] = 1
                onehot[i].append(OnehotPad)
            onehot[i] = np.array(onehot[i])
        return np.array(onehot)

def main(args):
    training_label    = args.file1
    testing_label     = args.file2
    training_features = args.file3
    testing_features  = args.file4
    word2vec          = args.model
    
    dataset = DataSet(word2vec, training_features, training_label, testing_features, testing_label)
    dataset.next_batch()

def parse():
    parser = ArgumentParser()
    parser.add_argument('--file1', '-f1',  default='MLDS_hw2_1_data/training_label.json', help='the file that contains texts')
    parser.add_argument('--file2', '-f2',  default='MLDS_hw2_1_data/testing_label.json', help='the file that contains texts')
    parser.add_argument('--file3', '-f3',  default='MLDS_hw2_1_data/training_data/id.txt', help='the file that contains training features')
    parser.add_argument('--file4', '-f4',  default='MLDS_hw2_1_data/testing_data/id.txt', help='the file that contains testing features')
    parser.add_argument('--model', '-m',  default='word_model/model.wv', help='the pretrained word2vec model')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())
