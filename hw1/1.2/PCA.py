import numpy as np
import pickle
import sys

def PCA():
    if len(sys.argv) != 3:
        print ('usage: python3.5 PCA.py mnist_exp_#(1~8)/Weights_dict.pickle layer_#(1~4,all)')
        sys.exit()
    with open(sys.argv[1], 'rb') as f:
        origin_weights = pickle.load(f)
    weights_1 = np.array(origin_weights['layer_1'])
    weights_2 = np.array(origin_weights['layer_2'])
    weights_3 = np.array(origin_weights['layer_3'])
    weights_4 = np.array(origin_weights['layer_4'])

    weights_1 = weights_1.reshape((weights_1.shape[0], -1))
    weights_2 = weights_2.reshape((weights_2.shape[0], -1))
    weights_3 = weights_3.reshape((weights_3.shape[0], -1))
    weights_4 = weights_4.reshape((weights_4.shape[0], -1))
    # print (weights_1.shape)
    # print (weights_2.shape)
    # print (weights_3.shape)
    # print (weights_4.shape)
    # sys.exit()
    weights_all = np.hstack((weights_1, weights_2))
    weights_all = np.hstack((weights_all, weights_3))
    weights_all = np.hstack((weights_all, weights_4))
    
    weights_1 = np.transpose(weights_1)
    weights_2 = np.transpose(weights_2)
    weights_3 = np.transpose(weights_3)
    weights_4 = np.transpose(weights_4)
    weights_all = np.transpose(weights_all)
    
    if sys.argv[2] == 'layer_1':
        weights = weights_1
        u, s, vh = np.linalg.svd(weights, full_matrices=False)
        tran_weights = np.transpose(weights)
        reduced_weights = np.dot(tran_weights, u[:, :2]).transpose()
        return reduced_weights
    elif sys.argv[2] == 'layer_2':
        weights = weights_2
        u, s, vh = np.linalg.svd(weights, full_matrices=False)
        tran_weights = np.transpose(weights)
        reduced_weights = np.dot(tran_weights, u[:, :2]).transpose()
        return reduced_weights
    elif sys.argv[2] == 'layer_3':
        weights = weights_3
        u, s, vh = np.linalg.svd(weights, full_matrices=False)
        tran_weights = np.transpose(weights)
        reduced_weights = np.dot(tran_weights, u[:, :2]).transpose()
        return reduced_weights
    elif sys.argv[2] == 'layer_4':
        weights = weights_4
        u, s, vh = np.linalg.svd(weights, full_matrices=False)
        tran_weights = np.transpose(weights)
        reduced_weights = np.dot(tran_weights, u[:, :2]).transpose()
        return reduced_weights
    elif sys.argv[2] == 'layer_all':   
        weights = weights_all
        u, s, vh = np.linalg.svd(weights, full_matrices=False)
        tran_weights = np.transpose(weights)
        reduced_weights = np.dot(tran_weights, u[:, :2]).transpose()
        return reduced_weights
    else:
        print ('usage: python3.5 PCA.py mnist_exp_#(1~8)/Weights_dict.pickle layer_#(1~4,all)')
        sys.exit()

weight = PCA()
filename = sys.argv[1].split('/')[0]+'_'+sys.argv[2]
print (filename)

import matplotlib.pyplot as plt
plt.plot(weight[0, :], weight[1, :], 'r.')
