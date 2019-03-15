# HW1-2 Optimization

* **Visualize the optimization process.**
* **Observe gradient norm during training.**
* **What happens when gradient is almost zero?**

## Visualize the optimization process using PCA

We use three hidden layers and a output layer to train our model for both mnist and simulating function. During the training, we dump all the weights from all four layers in directory ./hw1_2/func_exp_{} and ./hw1_2/mnist_exp_{} every 3 epochs in our eight experiments. There are 60 epochs total and thus will generate 20 lists of datas of weights after training. The data will be store in file Weights_dict.pickle in the above two directories.
We then use SVD to get the first two eigenvectors and reconstruct the weight metric to size (2,20).

## Observe gradient norm

Use the same model mentioned above and store the gradient norms in the same two directories as grad_norms.pickle.

### Training

To train the model, you only need to run run_func.sh and run_mnist.sh as well as specify the times of experiments.

```
./run_func.sh 1
./run_func.sh 3
```
Each trained parameters will be stored in directory ./hw1_2/func_exp_{} and ./hw1_2/mnist_exp_{}.

### Run PCA

Simply execute PCA.py and specify the model (mnist/func) as well as the layers you want to observe (layer1~4/layer_all). It will fetch all the weights of the specified layer from all eight layers and plot it on the chart. 

```
python3 PCA.py mnist layer_1
```
The result will be saved in directory ./hw1_2/Opt_Vis. 

## What happens when gradient is almost zero? (Simulate function sinc(3x))

First, train the network with original loss function 50 epochs. Then we change the objective function to gradient norm and keep training in order to figure out the parameters. We train this simulation model for 100 times and find each loss and minimal ratio when gradient is alomst zero.

### Library

* Uses python3
* Pytorch==1.0.1.post2
* matplotlib==2.2.3
* numpy==1.16.0

### Training

To train the model, you only need to use python3 to execute the simulate.py in ./hw1_2/simulate_function directory.

```
python3 simulate.py
```

Each trained parameters will be stored in directory ./hw1_2/models as gradient_norm_para_{}.pkl.

### Calculate Minimal Ratio and Visualization

minimal_ratio.py will load every model saved in ./hw1_2/models. After computing each minimal_ratio, it will plot the loss vs minimal_ratio as minimal_ratio_vs_loss.png.

```
python3 minimal_ratio.py
```


