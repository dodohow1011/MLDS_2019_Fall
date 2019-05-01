import os
import multiprocessing
from argparse import ArgumentParser
from gensim.models import Word2Vec, KeyedVectors

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


def main(args):
    corpus  = args.corpus
    dim     = args.dimension
    saveDir = args.save

    if not os.path.isdir(saveDir): os.mkdir(saveDir)

    savepath = os.path.join(saveDir, 'WordModel_{}.wv'.format(dim))

    sentences = []    

    with open(corpus, 'r') as f:
        for line in f:
            sentences.append(line.strip().split()) 

    cores = multiprocessing.cpu_count()
    
    word_model = Word2Vec( min_count=3, size=dim,sg=1,iter=30, workers=cores)
    word_model.build_vocab(sentences)
    word_model.train(sentences, total_examples=word_model.corpus_count, epochs=word_model.iter)
    word_model.init_sims(replace=True)
    word_model.wv.save(savepath)

def parse():
    parser = ArgumentParser()
    parser.add_argument('--corpus', '-c', required=True, help='the corpus file')
    parser.add_argument('--dimension', '-d', default=1024, type=int,  help='the latent dimension')
    parser.add_argument('--save', '-s',  required=True, help='the directory to save the trained model')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())
