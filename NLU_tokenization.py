#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os,re,math,sklearn
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings; warnings.filterwarnings(action='ignore')
import jieba

#========================================================
#  分词技术(规则分词)
#========================================================

class tokenization_FMM(object):
    '''
    正向最大匹配法
    '''
    def __init__(self, dic_path):
        self.dictionary = set()
        self.maximum = 0
        # 读取词典
        with open(dic_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                self.maximum = len(line)
    
    def cut(self, text):
        result = []
        index = 0
        while index < len(text):
            word = None
            for size in range(self.maximum, 0, -1):
                piece = text[index:(index+size)]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index += size
                    break
            if word is None:
                index += 1
        return result

class tokenization_BMM(object):
    '''
    逆向最大匹配法
    '''
    def __init__(self, dic_path):
        self.dictionary = set()
        self.maximum = 0
        # 读取词典
        with open(dic_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                self.maximum = len(line)
    
    def cut(self, text):
        result = []
        index = len(text)
        while index > 0:
            word = None
            for size in range(self.maximum, 0, -1):
                piece = text[(index-size):index]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index -= size
                    break
            if word is None:
                index -= 1
        return result
            
def tokenization_BM(dic_path, text):
    '''
    双向最大匹配法
    '''
    FMM = tokenization_FMM(dic_path)
    BMM = tokenization_BMM(dic_path)
    
    a = FMM.cut(text)
    b = BMM.cut(text)

    if a == b:
        return b
    lena = len(a)
    lenb = len(b)
    if lena >= lenb:
        return b
    if lena < lenb:
        return a

#========================================================
#  分词技术(统计分词)
#========================================================

class tokenization_HMM(object):
    '''
    隐马尔可夫模型分词
    '''
    def __init__(self):
        # 初始化成员信息和一些成员变量
        
        # 存储算法中间结果,不用每次都训练模型
        self.model_file = './data/hmm_model.pkl'
        # 状态值集合
        self.state_list = ['B','M','E','S']
        # 参数加载,用于判断是否重新加载model_file
        self.load_para = False

    def try_load_model(self, trained):
        # 用于加载已计算的中间结果,重新训练时需初始化清空结果

        if trained:
            import pickle
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.C_dic = pickle.load(f)
                self.load_para = True
        else:
            # 状态转移概率(状态->状态的条件概率)
            self.A_dic = {}
            # 发射概率(状态->词语的条件概率)
            self.B_dic = {}
            # 状态的初始概率
            self.pi_dic = {}
            self.load_para = False
    
    def train(self, path):
        # 用给定的分词语料进行训练,计算转移概率、发射概率、初始概率
        
        # 重置几个概率矩阵
        self.try_load_model(False)
        # 统计状态出现次数，求p(o)
        Count_dic = {}
        # 初始化参数
        def init_parameters():
            for state in self.state_list:
                self.A_dic[state] = {s:0.0 for s in self.state_list}
                self.B_dic[state] = {}
                self.pi_dic[state] = 0.0
                Count_dic[state] = 0.0

        def makeLabel(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B']+['M']*(len(text)-2)+['E']
            return out_text
        
        init_parameters()
        line_num = -1
        # 观察者集合,主要是字及标点等
        words = set()
        with open(path, encoding='utf8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                word_list = [i for i in line if i !=' ']
                words = set(word_list)  # 更新字的集合
                linelist = line.split()
                line_state = []
                for w in linelist:
                    line_state.extend(makeLabel(w))
                assert len(word_list)==len(line_state)
                for k,v in enumerate(line_state):
                    Count_dic[v] += 1
                    if k == 0:
                        self.pi_dic[v] += 1   # 每个句子的第一个字的状态,用于计算初始状态
                    else:
                        self.A_dic[line_state[k-1]][v] += 1  # 计算转移概率
                        self.B_dic[line_state[k]][word_list[k]] = self.B_dic[line_state[k]].get(word_list[k], 0) + 1.0 # 计算发射概率
        self.pi_dic = {k: v*1.0/line_num for k, v in self.pi_dic.items()}
        self.A_dic = {k: {k1: v1 / Count_dic[k] for k1, v1 in v.items()} for k, v in self.A_dic.items() }
        # 加1平滑
        self.B_dic = {k: {k1: (v1 + 1) / Count_dic[k] for k1, v1 in v.items()} for k,v in self.B_dic.items()}#序列化
        
        import pickle
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.pi_dic, f)
        return self

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        # viterbi算法的实现,一种动态规划的方法,主要是求最大概率的路径

        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]
        for t in range(1, len(text)):
            V.append({})
            newpath = {}
            # 检验训练的发射概率矩阵中是否有该字
            neverSeen = text[t] not in emit_p['S'].keys() and \
                text[t] not in emit_p['M'].keys() and \
                text[t] not in emit_p['E'].keys() and \
                text[t] not in emit_p['B'].keys()
            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0 #设置未知字单独成词
                (prob, state) = max(
                    [(V[t - 1][y0] * trans_p[y0].get(y, 0) *
                      emitP, y0)
                     for y0 in states if V[t - 1][y0] > 0])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])
            
        return (prob, path[state])

    def cut(self, text):
        # 切词
        import os
        if not self.load_para:
            self.try_load_model(os.path.exists(self.model_file))
        prob, pos_list = self.viterbi(text, self.state_list, self.pi_dic, self.A_dic, self.B_dic)
        begin, next = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i+1]
                next = i+1
            elif pos == 'S':
                yield char
                next = i+1
            print(next)
        if next < len(text):
            yield text[next:]
