import re
import os
from argparse import ArgumentParser

SEPERATION_FLAG = '+++$+++'

def main(args):
    corpus    = args.corpus
    outputDir = args.output

    if not os.path.isdir(outputDir): os.mkdir(outputDir)

    outputTrain  = os.path.join(outputDir, 'train.txt')
    outputLabel  = os.path.join(outputDir, 'label.txt')
    outputCorpus = os.path.join(outputDir, 'corpus.txt')

    fileTrain  = open(outputTrain, 'w')
    fileLabel  = open(outputLabel, 'w')
    fileCorpus = open(outputCorpus, 'w')

    datacount = 0
    with open(corpus, 'r') as f:
        pairs = []
        for line in f:
            line = line.strip()
            if line == SEPERATION_FLAG:
                paris = []
                continue
            pairs.append(line)
            fileCorpus.write('<bos> {} <eos>\n'.format(line))
            if len(pairs) < 2:
                continue;
            else:
                datacount += 1
                print ('\r{} data processed.'.format(datacount), end='', flush=True)
                fileTrain.write('<bos> {} <eos>\n'.format(pairs[0]))
                fileLabel.write('<bos> {} <eos>\n'.format(pairs[1]))
                pairs = pairs[1:]
    print()

def parse():
    parser = ArgumentParser()
    parser.add_argument('--corpus', '-c', required=True, help='the corpus file')
    parser.add_argument('--output', '-o', required=True, help='the output directory')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())
