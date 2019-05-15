# HW3-1 Image Generation

* **DCGAN**
* **WGAN**

## Library

* Uses python3
* Pytorch==1.0.1.post2
* numpy==1.16.0

## DCGAN

Our best model only uses the data from directory AnimeDataset/ which has 33431 images. However, you can add more datas from directory extra/ if you like.
For every epoch, we store the model and generate 25 anime faces in one png. 
The 25 generated image will be stored in directory dcgan_image/v2/.
The stored model will be in directory dcgan_model/v2

Usage: python3 main.py --mode train -e [number of epochs] -b [batch_size] -t [training data file] -m [directory to save model] -g [directory to store generated image] -l [directory to store loss]

```
python3 main.py --mode train -e 200 -b 64 -t train_image.txt -m dcgan_model/v2 -g dcgan_image/v2 -l loss_his
```

or

```
bash run.sh
```

