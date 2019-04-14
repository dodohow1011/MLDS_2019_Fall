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

## Schedule Sampling

We gradually increase the sampling rate from 0 to 0.9 along the training process.

### training

Usage: python3 --mode train -s [directory to save the trianed model]

```
python3 --mode train -s model_checkpoint_schedule_sampling_increase_rate/
```

### testing

Usage: python3 --mode inference --model [the trained model]

```
python3 --mode inference --model model_checkpoint_schedule_sampling_increase_rate/s2vt.pytorch
```

The result will be store in ./caption.txt

## S2VT & S2VT_v2 (with attention and schedule sampling)

### training

Usage: python3 s2vt.py/s2vt_v2.py --mode=train

```
python3 s2vt_v2.py --mode=train
```

The model will be stored in directory model/

### testing

Usage: python3 s2vt.py/s2vt_v2.py --mode=test --checkpoint=[checkpoint path] --output=[output file name]

```
python3 s2vt_v2.py --mode=test --checkpoint=model/s2vt_attn.pytorch --output=caption.txt
```

The result will be store in ./caption.txt

## Beam Search

We apply beam search to s2vt_v2.py. Since beam search is only concerned with the output mothed, we don't need to retrain the model. 

In this project, we use beam search with vocab_num as 3, group_num as 6. 

And the result is the sentence of the largest probability above all candidates answers.

Usage: python3 s2vt_v2_BS.py --mode [mode] --checkpoint [model_name] --output [output_filename]

```
python3 s2vt_v2_BS.py --mode inference --checkpoint ./model_checkpoint_embed_32/s2vt_att.pytorch --output
```
