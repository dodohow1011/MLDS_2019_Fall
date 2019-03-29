import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == '__main__':
    loss_list = []
    val_loss_list = []
    acc_list = []
    val_acc_list = []
    sen_list = []

    sen_list = np.load("norm_model.npy")
    batch_list = [16,32,64,128,256,512,1024,2048]
    record = np.load("loss_model.npy")
    for i in record:
        loss_list.append(i[0])
        acc_list.append(i[1])
        val_loss_list.append(i[2])
        val_acc_list.append(i[3])


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax1 = ax[0]

    ax1.set_xlabel('batch size')
    ax1.set_ylabel('loss', color = 'b')
    ax1.semilogx(batch_list, loss_list, color = 'b', label = 'train loss')
    ax1.semilogx(batch_list, val_loss_list, color = 'b', linestyle = '--', label = 'test loss')
    ax1.tick_params(axis='y', labelcolor= 'b')
    ax2 = ax1.twinx()
    ax2.set_ylabel('sensitivity', color = 'r')
    ax2.semilogx(batch_list, sen_list, color = 'r', label = 'sensitivity')
    ax2.tick_params(axis='y', labelcolor= 'r')
    ax1.legend(loc="best") 
    ax2.legend(loc="best")

    ax3 = ax[1]
    ax3.set_title('acc & sensitivity v.s. batch_size')
    ax3.set_xlabel('batch size')
    ax3.set_ylabel('acc', color = 'b')
    ax3.semilogx(batch_list, acc_list, color = 'b', label = 'train acc')
    ax3.semilogx(batch_list, val_acc_list, color = 'b', linestyle = '--', label = 'test acc')
    ax3.tick_params(axis='y', labelcolor= 'b')
    ax4 = ax3.twinx()
    ax4.set_ylabel('sensitivity', color = 'r')
    ax4.semilogx(batch_list, sen_list, color = 'r', label = 'sensitivity')
    ax4.tick_params(axis='y', labelcolor= 'r')
    ax3.legend(loc="upper left") 
    ax4.legend(loc="best")

    plt.tight_layout()
    fig.savefig('sensitivity.png')
