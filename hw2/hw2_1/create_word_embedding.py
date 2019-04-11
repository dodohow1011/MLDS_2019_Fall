import json
import re
import multiprocessing
from argparse import ArgumentParser
from gensim.models import Word2Vec, KeyedVectors

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


def main(args):
    filename1 = args.file1
    filename2 = args.file2
    dim       = args.dimension
    savepath  = args.save

    sentences = []    

    with open(filename1, 'r') as f:
        data = json.loads(f.read())
    for movie in data:
        sentences += [ ('<BOS> '+re.sub(',', '', x[:-1].lower())+' <EOS>').strip().split() for x in movie['caption']]

    with open(filename2, 'r') as f:
        data = json.loads(f.read())
    for movie in data:
        sentences += [ ('<BOS> '+re.sub(',', '', x[:-1].lower())+' <EOS>').strip().split() for x in movie['caption']]


    cores = multiprocessing.cpu_count()

    word_model = Word2Vec(min_count=3,
                          size=dim,
                          sg=1,
                          iter=30,
                          workers=cores)

    word_model.build_vocab(sentences)
    word_model.train(sentences, total_examples=word_model.corpus_count, epochs=word_model.iter)
    word_model.init_sims(replace=True)
    word_model.wv.save(savepath)

def parse():
    parser = ArgumentParser()
    parser.add_argument('--file1', '-f1',  default='MLDS_hw2_1_data/training_label.json', help='the file that contains texts')
    parser.add_argument('--file2', '-f2',  default='MLDS_hw2_1_data/testing_label.json', help='the file that contains texts')
    parser.add_argument('--dimension', '-d', default=256, type=int,  help='the latent dimension')
    parser.add_argument('--save', '-s',  required=True, help='the path to save the trained model')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())
