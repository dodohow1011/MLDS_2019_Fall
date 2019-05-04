import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import sys
import os

class Encoder(nn.Module):
    def __init__(self, batch_size, seq_size, word_emb_size , hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.word_emb_size = word_emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size=word_emb_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)

    def forward(self, input_seq):
        # initialize
        h = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)).cuda()
        output, h = self.gru(input_seq, h)
        return output, h

    def inference(self, input_seq):
        h = Variable(torch.zeros(self.n_layers, 1, self.hidden_size)).cuda()
        output, h = self.gru(input_seq, h)

        return output, h

class Decoder(nn.Module):
    def __init__(self, batch_size, hidden_size, n_layers, dropout):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.one_hot = nn.Linear(hidden_size, 97839)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)

    def forward(self, response, encoder_h):
        output, h = self.gru(response, encoder_h)

        return output, h

    def inference(self, pre_word, encoder_h):
        output, h = self.gru(pre_word, encoder_h)
        output = output.reshape((1, self.hidden_size))
        _, word_id = torch.max(self.one_hot(output), 1)
        return h, word_id
