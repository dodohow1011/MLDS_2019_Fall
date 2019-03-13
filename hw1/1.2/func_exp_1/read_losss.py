import pickle
import sys
import numpy as np

with open(sys.argv[1], 'rb') as f:
    loss = pickle.load(f)

print (len(loss))
