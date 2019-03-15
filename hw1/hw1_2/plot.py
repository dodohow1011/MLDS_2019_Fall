import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys

if len(sys.argv) != 2:
    print ('usage: python3 plot.py mnist/func')
    sys.exit()

grads = [sys.argv[1]+'_exp_'+str(i)+'/grad_norms.pickle' for i in range(1,9)]
loss = [sys.argv[1]+'_exp_'+str(i)+'/Loss.pickle' for i in range(1,9)]


for i in range(0,8):
    grad_filename = sys.argv[1]+'_grad_'+str(i+1)
    loss_filename = sys.argv[1]+'_loss_'+str(i+1)
    with open(grads[i], 'rb') as g:
        gf = pickle.load(g)   
    plt.plot(gf)
    plt.ylabel('grad')
    plt.savefig('Plot/'+grad_filename+'.png')

    with open(loss[i], 'rb') as l:
        lf = pickle.load(l)
    plt.plot(lf)
    plt.ylabel('loss')
    plt.savefig('Plot/'+loss_filename+'.png')



