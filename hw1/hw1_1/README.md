# HW1-1 Deep vs Shallow

* **Simulate a Function**
* **Train on Actual Task**

## Simulate sinc(3x) - Change Directory to simulate_function/

Three different models are trained to fit the function sinc(3x) with input range [0, 1].
These models differ in structure: 1 layer (shallow), 2 layers, 4 layers (deep), however, we fixed the amout of parameters to approximately 300 in order to analyze the performance.

* figures/ -> plots the training loss and the trained result which approximate the traget function
* models/ -> stores the trained models
* history/ -> stores the training history including loss and epoch

### Libraries

* Uses python3
* Keras==2.2.4
* tensorflow=1.13.1
* matplotlib==2.2.3
* h5py==2.9.0

### Training

To train the model, run simulate.py. An argument must be passed to specify what kind of structure to use - shallow/deep/two_layers.

```
python3 simulate.py shallow
```

The training history would then be stored into history as shallow_history.pickle.
Repeat the process but change the passed argument to obtain the result of two_layers and deep network.

### Visualizing Results

After the training process is completed, to plot the training loss and result, run plot_loss.py and reconstruct.py seperately.

```
python3 plot_loss.py history/shallow_history.pickle history/two_layers_history.pickle history/deep_history.pickle
python3 reconstruct.py models/shallow_model.h5 models/two_layers_model.h5 models/deep_model.h5
```

The figures would be stored in figures/ as training_loss.png and result.png which were demonstrated in the presentation.

The models and histories are already saved, so running this part is enough to reproduce our results. A shell script - visualize.sh - is provided for simplification.

```
./visualize.sh
```

However, if you wish to train the three different networks all over again, simply run run.sh.

```
./run.sh
```

## Train on MNIST - Change Directory to train_on_mnist/

Three different models are trained on the mnist dataset.
These models differ in structure: 2 layer (shallow), 4 layers, 6 layers (deep), however, we fixed the amout of parameters to approximately 2600 in order to analyze the performance.

* figures/ -> plots the training loss and the training accuracy
* models/ -> stores the trained models
* history/ -> stores the training history including loss accuracy and epoch

### Libraries

Uses python3
Keras==2.2.4
tensorflow=1.13.1
matplotlib==2.2.3
h5py==2.9.0

### Training

To train the model, run mnist_keras.py. An argument must be passed to specify what kind of structure to use - deep2/deep4/deep6.

```
python3 mnist_keras.py deep2
```

The training history would then be stored into history as deep2_history.pickle.
Repeat the process but change the passed argument to obtain the results of deep4 and deep6 network.

### Visualizing Results

After the training process is completed, to plot the training loss and result, run plot_loss.py and plot_acc.py seperately.

```
python3 plot_loss.py history/deep2_history.pickle history/deep4_history.pickle history/deep6_history.pickle
python3 plot_acc.py history/deep2_history.pickle history/deep4_history.pickle history/deep6_history.pickle
```

The figures would be stored in figures/ as training_loss.png and training_acc.png which were demonstrated in the presentation.

The models and histories are already saved, so running this part is enough to reproduce our result. A shell script - visualize.sh - is provided for simplification.

```
./visualize.sh
```

However, if you wish to train the three different networks all over again, simply run run.sh.

```
./run.sh
```
