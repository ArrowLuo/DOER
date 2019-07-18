# DOER

The implementation of 

*[DOER: Dual Cross-Shared RNN for Aspect Term-Polarity Co-Extraction](https://arxiv.org/pdf/1906.01794.pdf). 
Huaishao Luo, Tianrui Li, Bing Liu, Junbo Zhang. ACL, 2019.*

This paper focuses on two related subtasks of aspect-based sentiment analysis, namely **aspect term extraction (ATE)** 
and **aspect sentiment classification (ASC)**, which we call aspect term-polarity co-extraction.

## Requirements

* python 2.7
* tensorflow==1.2.0

```
pip install -r requirements.txt
```

## Running

##### preprocess

```sh
python main.py --do_preprocess --data_sets laptops_2014
```

##### train

```sh
python main.py \
    --do_train --do_evaluate \
    --data_sets laptops_2014 \
    --choice_rnncell regu \
    --use_mpqa \
    --use_labels_length \
    --do_cross_share --lr 0.001 \
    --batch_size 16
```

See [main.py](./main.py) for more training arguments.

## Citation

If this work is helpful, please cite as:

```
@Inproceedings{Luo2019doer,
    author = {Huaishao Luo and Tianrui Li and Bing Liu and Junbo Zhang},
    title = {DOER: Dual Cross-Shared RNN for Aspect Term-Polarity Co-Extraction},
    booktitle = {ACL},
    year = {2019}
}
```
