# ADK-KG
This is an implementation of the paper [Adapting Distilled Knowledge for Few-shot Relation Reasoning over Knowledge Graphs].


## Disclaimer
Please note that this is highly-experimental research code, not well documented and we provide no warranty of any kind. Use at your own risk!

## Requirements
- python3 (3.8.3)
- pytorch (1.7.1)

## Data Preparation

There are three datasets under folder `data`. In our experiments, dataset could be NE, fb15k-237, and wiki.

``` bash
data/FB15K-237
data/NE
data/wiki
```
## Pretrain text embedding for entities and relation based on Bert:

``` bash
text_emb/bert/sentence-transformers/examples/applications/computing-embeddings/embeding2.py
```

## Train

``` bash
experiments.py --data_dir <data dir> --model <model name> --train
```

## Test

``` bash
experiments.py --data_dir <data dir> --model <model name> --few_shot
```
