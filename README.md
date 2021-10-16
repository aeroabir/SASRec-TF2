# SASRec: Self-Attentive Sequential Recommendation

Based on the paper

> Wang-Cheng Kang, Julian McAuley (2018). Self-Attentive Sequential Recommendation. In Proceedings of IEEE International Conference on Data Mining (ICDM'18)

https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf

Modified the original Git Repo: https://github.com/kang205/SASRec and the TF 2.x version: https://github.com/nnkkmto/SASRec-tf2

## Datasets

All the Amazon product datasets can be downloaded using the script download_and_process_amazon.py

## Algorithms

1. SASRec: Original Transformer based Recommender
2. SSEPT: Transformer with User embeddings
3. SASRec++: Transformer with item embeddings learnt from GCN
4. TiSASRec: Time Interval aware SASRec
5. RNN: RNN based sequence prediction
6. HSASRec: Hierarchical SASRec - using previous user history embeddings


## Usage

 1. python main-tf2.py --dataset=ae --train_dir=default --maxlen=50 --dropout_rate=0.5 --lr=0.001 --hidden_units=100 --num_epochs=50 --text_features=1 (for SASRec)
 2. python main-tf2.py --dataset=ae_v2 --train_dir=default --maxlen=200 --dropout_rate=0.5 --lr=0.001 --hidden_units=100 --num_epochs=50 --add_embeddings=1 (for SASRec++)
 3. python main-tf2.py --dataset=ae_v2 --train_dir=default --maxlen=200 --dropout_rate=0.5 --lr=0.001 --hidden_units=100 --num_epochs=50 --model_name=ssept (for SSEPT)
 4. python main-tf2.py --dataset=ae_graph --train_dir=default --maxlen=50 --dropout_rate=0.5 --lr=0.001 --hidden_units=100 --num_epochs=50 --add_history=1 --model_name=hsasrec


