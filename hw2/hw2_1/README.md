# HW2-1 Video Caption Generation

* **Pretrain Word Embedding**
* **s2vt**
* **Attention**
* **Schedule Sampling**
* **Beam Search**

## Library

* Uses python3
* Pytorch==1.0.1.post2
* numpy==1.16.0
* gensim==2.7.1

## Word Embedding

Instead of training the embedding layer with the whole RNN network, we pretrained a embedding network for the training corpus.

We used sentences in the training labels and testing labels to build this word2vec model.

Usage: python3 create\_word\_model.py -f1 [training labels] -f2 [testing labels] -d [embedding dimension] -s [path to save model]

```
python3 create_word_model.py -f1 MLDS_hw2_1_data/training_label.json -f2 MLDS_hw2_1_data/testing_label.json -d 256 -s word_model/model.wv
```

## s2vt

## Attention

## Schedule Sampling



## Beam Search

### Training

To train models with different amout of paramters, run random\_mnist.py and specify the number of filters, device and patience which controls when to stop the training process.

```
python3 random_mnist.py 0 7 300
```

The result will be stored in model\_{filters} directory.
The example above will train on mnist with random labels on GPU 0 with 300 filters on each convolution layers.
The training process will stop if accuracy does not improve in 7 epochs.
To visualize the comparison between losses and number of parameters, run plot\_result.py.

```
python3 plot_result.py <gpuID>
```

The figures will be saved as Model\_Loss.png and Model\_Acc.png in ./

## Number of parameters v.s. Generalization

## Flatness v.s. Generalization

Train on the same model with different approach. We compared the result by changing batch size and learning rate respectively.

### Training

Enter the corresponding directory.

```
cd interpolation
```

To train the model, run mnist.py and set the batch size and learning rate.

```
python3 mnist.py <batch size> <learning rate> <gpuID>
python3 mnist.py 64 1e-4 0
```

The example above will train mnist with batch size 64, learing rate 0.0001 on the GPU whose ID is 0.

The trained model will be saved in batchsize\_{}\_lr\_{}/

### Interpolation and Plot

Create a directory first to store the interpolation result.

```
mkdir batchsize_comparison
```

And then run interpolation.py, specifying the two models to interpolate and the directory to store the result.

```
python3 interpolate.py <model_1> <model_2> <gpuID> <output directory>
python3 interpolate.py batchsize_{}_lr_{} batchsize_{}_lr_{} 0 batchsize_comparison
```

A pickled file would be stored in batchsize\_comparison/.

To plot the result, run plot.py and specify the file produced previously

```
python3 plot.py batchsize_comparison/result.pkl
```

The figure will be saved in the same directory where result.pkl is stored.
