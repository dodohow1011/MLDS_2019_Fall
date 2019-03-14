import pickle
import sys
import numpy as np

if len(sys.argv) != 2:
    print ('usage: python3 read_grad.py <gradient file>')
    sys.exit()
with open(sys.argv[1], 'rb') as f:
    loss = pickle.load(f)

print (len(loss))
