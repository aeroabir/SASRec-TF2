# SASRec: Self-Attentive Sequential Recommendation

行動系列に対するレコメンデーションに関する以下論文の TensorFlow2.0での実装

> Wang-Cheng Kang, Julian McAuley (2018). Self-Attentive Sequential Recommendation. In Proceedings of IEEE International Conference on Data Mining (ICDM'18)

https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf

## Datasets

> “The Instacart Online Grocery Shopping Dataset 2017”, Accessed from https://www.instacart.com/datasets/grocery-shopping-2017 on 2019/09/16

https://www.instacart.com/datasets/grocery-shopping-2017

Instacartの購買データセット


## Usage

### データ準備

上記urlからデータセットをダウンロード

data/packed 以下に配置

`python instacart_preprocess.py`

で元データ準備

### 学習

`python train.py`

# SASRec-TF2
