#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np 
import os,re,math,sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings; warnings.filterwarnings(action='ignore')
from tools import docs2tfidf_sk

#========================================================
#  文本聚类
#========================================================

def Cluster_kmeans(docs, numClass):
    '''
    中文文本聚类(基于kmeans)
    INPUT  -> 文本集, 聚类数量
    OUTPUT -> 训练好的聚类模型
    '''
    tfidf = docs2tfidf_sk(docs)
    pca = PCA(n_components=10)  # 降维
    tfidf_new = pca.fit_transform(tfidf)

    model = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)
    trained_model = model.fit(tfidf_new)

    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    # print("估簇的个数是否合适:", trained_model.inertia_)

    # 中心点
    # print("中心点：", trained_model.cluster_centers_)
    # 绘制中心点
    # center_x = []
    # center_y = []
    # for point in trained_model.cluster_centers_:
    #     try:
    #         center_x.append(point[0])
    #         center_y.append(point[1])
    #     except:
    #         pass
    # plt.plot(center_x, center_y, "rv")
    # plt.show()

    return trained_model