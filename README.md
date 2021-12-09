# University Course Recommnder System using R-GCN & Glove

* A python implementation of course recommender system using R-GCN and GloVe
* This model is created based on the students' feedback that they would like to refelct their course preferences to the result


## Demo
<img src="https://github.com/Jennie1021/RGCN-GLOVE_course_recommender/blob/main/img/AI%EC%84%A0%EB%B0%B0_%EA%B5%90%EC%96%91%EC%B6%94%EC%B2%9C.gif?raw=true" width="650" height="500"/>



## Knowlege Graph Generation(example)
<img src="https://github.com/Jennie1021/RGCN-GLOVE_course_recommender/blob/main/img/KG.png?raw=true" width="650" height="400"/>

## Algorithm Logic
<img src="https://github.com/Jennie1021/RGCN-GLOVE_course_recommender/blob/main/img/RGCN_GloVe_Course_Recommender_Logic.jpg?raw=true" width="650" height="400"/>


## RGCN implementation
```
python3 utils.py
python3 model.py
python3 main.py --n-epochs 100000 --evaluate-every 500 --graph-batch-size 45000
```

## Glove implementation
```
python3 utils.py
python3 main.py
```

## Final implementation of recommendation 
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
