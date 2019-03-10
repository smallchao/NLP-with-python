#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np 
import os,re,sklearn
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB  # 参数存在负值用GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import xgboost as xgb
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings; warnings.filterwarnings(action='ignore')
from tools import doc2hashing_sk

#========================================================
#  文本分类
#========================================================

def classifier_nb(docs, labels):
    '''
    中文文本分类(基于nb)
    INPUT  -> 文本集, 标签
    OUTPUT -> 训练好的分类模型
    '''
    HashCode = []
    for doc in doc:
        HashCode.append(doc2hashing_sk(docs))

    x_train, x_valid, y_train, y_valid = train_test_split(HashCode, labels, test_size=0.10)

    nb_model = MultinomialNB()
    nb_model.fit(x_train, y_train)

    print(nb_model.score(x_valid, y_valid))

    return nb_model

def classifier_svm(docs, labels):
    '''
    中文文本分类(基于svm)
    INPUT  -> 文本集, 标签
    OUTPUT -> 训练好的分类模型
    '''
    HashCode = []
    for doc in doc:
        HashCode.append(doc2hashing_sk(docs))

    x_train, x_valid, y_train, y_valid = train_test_split(HashCode, labels, test_size=0.10)

    svm_model = SVC(kernel='linear')
    svm_model.fit(x_train, y_train)

    print(svm_model.score(x_valid, y_valid))

    return svm_model

def classifier_xgb(docs, labels):
    '''
    中文文本分类(基于xgb)
    INPUT  -> 文本集, 标签
    OUTPUT -> 训练好的分类模型
    '''
    HashCode = []
    for doc in doc:
        HashCode.append(doc2hashing_sk(docs))

    x_train, x_valid, y_train, y_valid = train_test_split(HashCode, labels, test_size=0.10)

    dtrain = xgb.DMatrix(x_train, y_train)
    dvalid = xgb.DMatrix(x_valid, y_valid)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    # 建模参数
    params = {
            "booster" : "gbtree",
            "objective": "multi:softmax",  # 多分类的问题
            # "objective": "multi:softprob",  # 多分类概率
            # "objective": "binary:logistic",  # 二分类
            # "objective": "reg:linear",  # 回归问题
            'eval_metric': 'merror',  # logloss
            'num_class': 2, # 类别数,与multi:softmax并用
            'gamma': 0.1,  # 用于控制是否后剪枝的参数，越大越保守，一般0.1、0.2
            'max_depth': 8,   # 构建树的深度，越大越容易过拟合
            'alpha': 0, # L1正则化系数
            'lambda': 10, # 控制模型复杂度的权重值的L2正则化项参数,参数越大越不容易过拟合
            'subsample': 0.8,   # 随机采样训练样本 训练实例的子采样比
            'colsample_bytree': 0.5,  # 生成树时进行列采样
            'min_child_weight': 3,  
            'silent': 0, # 设置成1则没有运行信息输出
            'eta': 0.03,  # 学习率
            'nthread':-1, # cpu线程数
            # 'n_estimators': 300,
            "seed": 10   # 随机种子
            }

    xgb_model = xgb.train(params, dtrain, 1000, evals=watchlist, early_stopping_rounds=300, verbose_eval=True)

    # 保存模型
    xgb_model.save_model(FILE_DIR+'\\xgb_weight.model')
    
    return xgb_model
