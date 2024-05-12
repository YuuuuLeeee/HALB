# HALB: Hierarchy-Aware and Label Balanced Model for Hierarchical Text Classification

This repository implements a model with multi-label negative supervision and asymmetric loss (ASL) for hierarchical text classification. 
This work is for the long paper "Hierarchy-Aware and Label Balanced Model for Hierarchical Text Classification".

## Requirements

* Python >= 3.6
* torch >= 1.6.0
* transformers == 4.2.1
* fairseq >= 0.10.0

## Preprocess

Please download the original dataset and then use these scripts.

### WebOfScience

The original dataset can be acquired in [the repository of HDLTex](https://github.com/kk7nc/HDLTex). Preprocess code could refer to [the repository of HiAGM](https://github.com/Alibaba-NLP/HiAGM) and we provide a copy of preprocess code here.

```shell
cd ./data/WebOfScience
python preprocess_wos.py
python data_wos.py
```

### NYT

The original dataset can be acquired [here](https://catalog.ldc.upenn.edu/LDC2008T19).

```shell
cd ./data/nyt
python data_nyt.py
```

### RCV1-V2

The preprocess code could refer to the [repository of reuters_loader](https://github.com/ductri/reuters_loader) and we provide a copy here. The original dataset can be acquired [here](https://trec.nist.gov/data/reuters/reuters.html) by signing an agreement.

```shell
cd ./data/rcv1
python preprocess_rcv1.py
python data_rcv1.py
```

### Amazon

The original dataset can be acquired [here](https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification).

```shell
cd ./data/amazon
python preprocess_amazon.py
python data_amazon.py
```

## Train

```
usage: train.py [-h] [--lr LR] [--data {WebOfScience,nyt,rcv1,amazon}] [--batch BATCH] [--early-stop EARLY_STOP] [--device DEVICE] --name NAME [--update UPDATE] [--warmup WARMUP] [--contrast CONTRAST] [--graph GRAPH] [--layer LAYER]
                [--multi] [--lamb LAMB] [--thre THRE] [--tau TAU] [--NSL NSL] [--gama_neg GAMA_NEG] [--gama_pos GAMA_POS] [--seed SEED] [--wandb]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate.
  --data {WebOfScience,nyt,rcv1}
                        Dataset.
  --batch BATCH         Batch size.
  --early-stop EARLY_STOP
                        Epoch before early stop.
  --device DEVICE		cuda or cpu. Default: cuda
  --name NAME           A name for different runs.
  --update UPDATE       Gradient accumulate steps
  --warmup WARMUP       Warmup steps.
  --contrast CONTRAST   Whether use contrastive model. Default: True
  --graph GRAPH         Whether use graph encoder. Default: True
  --layer LAYER         Layer of Graphormer.
  --multi               Whether the task is multi-label classification. Should keep default since all 
  						datasets are multi-label classifications. Default: True
  --lamb LAMB           Weight of contrastive learning.
  --thre THRE           Threshold for keeping tokens. Denote as gamma in the paper.
  --tau TAU             Temperature for contrastive model.
  --nsl NSL             Weight of negative supervision.
  --gama_neg GAMA_NEG   gamma_neg of ASL.
  --gama_pos GAMA_POS   gamma_pos of ASL.
  --seed SEED           Random seed.
  --wandb               Use wandb for logging.
```

Checkpoints are in `./checkpoints/DATA-NAME`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively 
(`checkpoint_best_macro.pt`, `checkpoint_best_micro.pt`).

e.g. Train on `WebOfScience` with `batch=5, lr=1e-5, lambda=0.01, thre=0.02, nsl=0.005`. Checkpoints will be in `checkpoints/WebOfScience-test/`.

```shell
python train.py --name test --batch 5 --data WebOfScience --lambda 0.01 --thre 0.02 --nsl 0.005
```


We experiment on GeForce RTX 3000 (10G) with CUDA version $11.3$.

* The following settings can achieve higher results with unfixed seed (which we reported in the paper) .
```
WOS: lambda 0.01 thre 0.02 nsl=0.005 gamma_neg=2 gamma_pos=1
RCV1: lambda 0.01 thre 0.005 nsl=0.005 gamma_neg=3 gamma_pos=2
NYT: lambda 0.03 thre 0.005 nsl=0.01 gamma_neg=3 gamma_pos=1
Amazon: lambda 0.01 thre 0.002 nsl=0.005 gamma_neg=2 gamma_pos=1
```

* We keep the tau to $1$ for simplicity.

## Test

```
usage: test.py [-h] [--device DEVICE] [--batch BATCH] --name NAME [--extra {_macro,_micro}]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --batch BATCH         Batch size.
  --name NAME           Name of checkpoint. Commonly as DATA-NAME.
  --extra {_macro,_micro}
                        An extra string in the name of checkpoint. Default: _macro
```

Use `--extra _macro` or `--extra _micro`  to choose from using `checkpoint_best_macro.pt` or`checkpoint_best_micro.pt` respectively.

e.g. Test on previous example.

```shell
python test.py --name WebOfScience-test
```
