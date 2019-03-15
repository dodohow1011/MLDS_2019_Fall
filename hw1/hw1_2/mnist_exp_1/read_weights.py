import pickle
import sys
import numpy as np

with open(sys.argv[1], 'rb') as f:
    weights = pickle.load(f)

print (weights['layer_1'].shape)
print (weights['layer_2'].shape)
print (weights['layer_3'].shape)
print (weights['layer_4'].shape)
