import sys
import math
import numpy as np
from keras.models import load_model

shallow    = sys.argv[1]
two_layers = sys.argv[2]
deep       = sys.argv[3]

shallow_model    = load_model(shallow)
two_layers_model = load_model(two_layers)
deep_model       = load_model(deep)

sample_x = np.arange(0, 1, 0.001).reshape((-1, 1))

shallow_y    = shallow_model.predict(sample_x).reshape(-1)
two_layers_y = two_layers_model.predict(sample_x).reshape(-1)
deep_y       = deep_model.predict(sample_x).reshape(-1)

sample_x = sample_x.reshape(-1)
ground_truth = np.array([math.sin(3*math.pi*x)/(3*math.pi*x) for x in sample_x]).reshape(-1)

import matplotlib.pyplot as plt

plt.plot(sample_x, shallow_y)
plt.plot(sample_x, two_layers_y)
plt.plot(sample_x, deep_y)
plt.plot(sample_x, ground_truth)
plt.legend(['single layer', 'two layers', 'four layers', 'target function'])
plt.savefig('figures/result.png')
