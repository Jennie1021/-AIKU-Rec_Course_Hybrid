# University Course Recommnder System using R-GCN & Glove

* A python implementation of course recommender system using R-GCN and GloVe
* Students can reflect their preference to the recommendation result by adding(clicking) course lists


## RGCN implementation(Terminal)
```
python3 utils.py
python3 model.py
python3 main.py --n-epochs 100000 --evaluate-every 500 --graph-batch-size 45000
```

## Glove implementation(Terminal)
```
python3 utils.py
python3 main.py
```

## Final recomender(Python3)
```
>>> from recommend import *
>>> rec = Recommend()
>>> rec.course_rec(student id, clicked course list)
```


## Requirements
* CUDA 10.1
* torch==1.6.0
* torch-geometric==1.7.0

## Reference
https://github.com/MichSchli/RelationPrediction   
https://aclanthology.org/D14-1162.pdf

## Data
num_entity: 41952
num_relation: 24
num_train_triples: 748751
num_valid_triples: 8956
num_test_triples: 27625

