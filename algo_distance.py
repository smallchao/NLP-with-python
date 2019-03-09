#-*- coding:utf-8 -*-
import os,math,json,hashlib
import numpy as np
from scipy.spatial.distance import pdist
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

#========================================================
#  距离/相似度计算
#========================================================

def manhattan_dis(p, q):
    ''' 
    曼哈顿距离/绝对距离(L1范数)
    INPUT  -> 长度一致的向量1、向量2
    举例: p = [1,2,6]; q = [1,3,5]
    '''
    p = np.mat(p)
    q = np.mat(q)
    # return np.linalg.norm(p-q, ord=1)
    return np.sum(np.abs(p-q))

def euclidean_dis(p, q):
    ''' 
    欧式距离(L2范数)
    INPUT  -> 长度一致的向量1、向量2
    '''
    p = np.mat(p)
    q = np.mat(q)
    # return math.sqrt(np.sum(np.square(p-q)))
    # return np.linalg.norm(p-q)
    return math.sqrt(np.sum(np.power(p-q, 2))) 

def euclidean_dis_2(p, q):
    ''' 
    坐标点间欧式距离(速度快)
    INPUT  -> 坐标1、坐标2
    '''
    return math.sqrt(math.pow(p[0]-q[0], 2)+math.pow(p[1]-q[1], 2))

def standardized_euclidean_dis(p, q):
    ''' 
    标准化欧式距离
    INPUT  -> 长度一致的向量1、向量2
    '''
    sumnum = 0
    for i in range(len(p)):
        # 计算si 分量标准差
        avg = (p[i]-q[i])/2
        si = math.sqrt((p[i]-avg)**2+(q[i]-avg)**2)
        sumnum += ((p[i]-q[i])/si )**2
    return math.sqrt(sumnum)

def chebyshev_dis(p, q):
    ''' 
    切比雪夫距离(L-∞范数)
    INPUT  -> 长度一致的向量1、向量2
    '''
    p = np.mat(p)
    q = np.mat(q)
    # return np.abs(p-q).max()
    # return np.linalg.norm(p-q, ord=np.inf)
    return np.max(np.abs(p-q))

def hanming_dis(p, q):
    ''' 
    汉明距离
    INPUT  -> 长度一致的向量1、向量2
    '''
    p = np.mat(p)
    q = np.mat(q)
    smstr = np.nonzero(p-q)
    return np.shape(smstr)[1]

def jaccard_dis(p, q):
    ''' 
    计算两个向量的杰卡德距离
    INPUT  -> 长度一致的向量1、向量2
    '''
    set_p = set(p)
    set_q = set(q)
    dis = float(len((set_p | set_q)-(set_p & set_q)))/ len(set_p | set_q)
    # dis = pdist([p, q],'jaccard')
    return dis

def jaccard_sim(p, q):
    ''' 
    计算两个向量的杰卡德相似系数
    INPUT  -> 长度一致的向量1、向量2
    '''
    set_p = set(p)
    set_q = set(q)
    sim = float(len(set_p & set_q))/ len(set_p | set_q)
    return sim

def cosin_dis(p, q):
    ''' 
    计算两个向量的余弦距离
    INPUT  -> 长度一致的向量1、向量2
    '''
    # return 1-np.dot(p,q)/(np.linalg.norm(p)*(np.linalg.norm(q)))
    return pdist(np.vstack([p, q]), 'cosine')[0]

def cosin_sim(p, q):
    ''' 
    计算两个向量的余弦相似度
    INPUT  -> 长度一致的向量1、向量2
    '''
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(p, q):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA==0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)