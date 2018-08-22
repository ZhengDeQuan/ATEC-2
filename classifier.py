#/usr/bin/env python
#coding=utf-8
from gcforest.gcforest import GCForest
import pickle
from sklearn.metrics.classification import classification_report
from sklearn.ensemble import RandomForestClassifier
import sys
import preprocess
import numpy as np
import random
import os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier","class_weight":"balanced", "n_estimators": 50, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression","class_weight":"balanced","C":0.1})
    #ca_config["estimators"].append({"n_folds": 5, "type": "SGDClassifier", "class_weight": "balanced","n_jobs": -1})

    config["cascade"] = ca_config
    return config

if __name__=='__main__':
    file_src_dict = {'embedding_file': './data/word_embedding.pkl', 'train_file': './data/train.pkl',
                     'evaluate_file': './data/val.pkl'}
    with open(file_src_dict['evaluate_file'], 'rb') as f:
        val_q, val_r, val_labels = pickle.load(f,encoding='iso-8859-1')
    val_fea=pickle.load(open('./data/train_fea.pkl','rb'),encoding='iso-8859-1')
    val_data=[[d,l] for d,l in zip(val_fea,val_labels)]
    random.shuffle(val_data)
    val_fea=[d[0] for d in val_data]
    val_labels=[d[1] for d in val_data]
    val_fea_1=[]
    val_fea_0=[]
    for i in range(0,len(val_labels)):
        if val_labels[i]==0:
            val_fea_0.append(val_fea[i])
        else:
            val_fea_1.append(val_fea[i])
    test_fea=val_fea_1[:int(len(val_fea_1)/2)]+val_fea_0[:int(len(val_fea_0)/2)]
    test_labels=[1]*int(len(val_fea_1)/2)+[0]*int(len(val_fea_0)/2)
    train_fea=val_fea_1[int(len(val_fea_1)/2):]*1+val_fea_0[int(len(val_fea_0)/2):]*1
    train_labels=[1]*(len(val_fea_1)-int(len(val_fea_1)/2))*1+[0]*(len(val_fea_0)-int(len(val_fea_0)/2))*1
    train_data=[[t,l] for t,l in zip(train_fea,train_labels)]
    test_data=[[d,l] for d,l in zip(test_fea,test_labels)]
    random.shuffle(train_data)
    random.shuffle(test_data)
    test_fea=[d[0] for d in test_data]
    test_labels=[d[1] for d in test_data]
    train_fea=[d[0] for d in train_data]
    train_labels=[d[1] for d in train_data]
    gc = GCForest(get_toy_config())  # should be a dict
    X_train_enc = gc.fit_transform(np.array(train_fea), np.array(train_labels))
    i = 0
    while os.path.exists('./gcForest_model/' + str(i)):
        i += 1
    os.makedirs('./gcForest_model/' + str(i))
    #pickle.dump(gc,open('./gcForest_model/'+ str(i)+'/model.pkl','wb+'),protocol=True)
    y_pred = gc.predict(np.array(test_fea))
    print(classification_report(test_labels,y_pred))
