import matplotlib.pyplot as plt
import pickle
import math

# with open('batchsize_comparison/result.pkl', 'rb') as f:
#     result = pickle.load(f)
with open('lr_comparison/result.pkl', 'rb') as f:
    result = pickle.load(f)

alpha = []
train_acc = []
train_loss = []
test_acc = []
test_loss = []

for a, trl, tra, tel, tea in result:
    alpha.append(a)
    train_acc.append(tra)
    train_loss.append(math.log10(trl))
    test_acc.append(tea)
    test_loss.append(math.log10(tel))

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('alpha')
ax1.set_ylabel('cross entropy loss (log scale)', color=color)
ax1.plot(alpha, train_loss, 'b-')
ax1.plot(alpha, test_loss, 'b--')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('accuracy', color=color)
ax2.plot(alpha, train_acc, 'r-')
ax2.plot(alpha, test_acc, 'r--')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Interpolation')
plt.legend(['train', 'test'])
plt.show()
