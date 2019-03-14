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

acc1 = history1['acc']
acc2 = history2['acc']
acc3 = history3['acc']

#######################################################################
#                         plot training loss                          #
#######################################################################

import matplotlib.pyplot as plt

plt.plot(acc1)
plt.plot(acc2)
plt.plot(acc3)
plt.title('training accuracy (CNN_on_mnist)')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['two layers', 'four layers', 'six layers'])
plt.savefig('figures/training_acc.png')
