# HW1-2 Optimization

* **Visualize the optimization process.**
* **Observe gradient norm during training.**
* **What happens when gradient is almost zero?**

## What happens when gradient is almost zero? (Simulate function sinc(3x))

First, train the network with original loss function 50 epochs. Then we change the objective function to gradient norm and keep training in order to figure out the parameters. We train this simulation model for 100 times and find each loss and minimal ratio when gradient is alomst zero.

### Library

* Uses python3
* Pytorch==1.0.1.post2
* matplotlib==2.2.3
* numpy==1.16.0

### Training

To train the model, you only need use python3 to execute the simulate.py in ./hw1_2/simulate_function directory.

```
python3 simulate.py
```

Each trained parameters will be stored in directory ./hw1_2/models as gradient_norm_para_{}.pkl.

### Calculate Minimal Ratio and Visualization

minimal_ratio.py will load every model saved in ./hw1_2/models. After computing each minimal_ratio, it will plot the loss vs minimal_ratio as minimal_ratio_vs_loss.png.

```
python3 minimal_ratio.py
```


