import sys
import pickle

inputfile1 = sys.argv[1]
inputfile2 = sys.argv[2]
inputfile3 = sys.argv[3]

with open(inputfile1, 'rb') as f:
    history1 = pickle.load(f)
with open(inputfile2, 'rb') as f:
    history2 = pickle.load(f)
with open(inputfile3, 'rb') as f:
    history3 = pickle.load(f)

loss1 = history1['loss']
loss2 = history2['loss']
loss3 = history3['loss']

#######################################################################
#                         plot training loss                          #
#######################################################################

import matplotlib.pyplot as plt

plt.plot(loss1)
plt.plot(loss2)
plt.plot(loss3)
plt.title('training loss (MSE)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['single layer', 'two layers', 'four layers'])
plt.savefig('training_loss.png')
