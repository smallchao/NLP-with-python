#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np 
import os,re,math,sklearn,random
from gensim import corpora
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings; warnings.filterwarnings(action='ignore')
from tools import doc2words

#========================================================
#  关键词提取
#========================================================

def keyword_TF(doc, topK=10):
    '''
    关键词提取(基于高频词)
    INPUT  -> 输入文本, 关键词数量
    OUTPUT -> 文本关键词
    '''
    wordlist = doc2words(doc, False)
    tf_dic = {}
    for word in wordlist:
        tf_dic[word] = tf_dic.get(word, 0) +1
    return sorted(tf_dic.items(), key=lambda x:x[1], reverse=True)[:topK]

def keyword_TFIDF(doc, n=5, stop_word_file_path=False):
    '''
    关键词提取(基于TF-IDF算法)
    INPUT  -> 输入文本, 关键词数量, 停用词文件地址
    OUTPUT -> 文本关键词
    '''
    if stop_word_file_path:
        jieba.analyse.set_stop_words(stop_word_file_path)
    keywords = jieba.analyse.extract_tags(doc, 
                                          topK=n, # 返回TF/IDF权重最大的关键词个数
                                          withWeight=False,  # 是否需要返回关键词权重值
                                          allowPOS=('n','nr','ns')  # 仅包括指定词性的词
                                          )
    result = " ".join(keywords)
    return result

def keyword_TextRank(doc, n=5, stop_word_file_path=False):
    '''
    关键词提取(基于TextRank算法)
    INPUT  -> 输入文本, 关键词数量, 停用词文件地址
    OUTPUT -> 文本关键词
    '''
    if stop_word_file_path:
        jieba.analyse.set_stop_words(stop_word_file_path)
    keywords = jieba.analyse.textrank(doc, 
                                      topK=n,
                                      withWeight=False,  # 是否需要返回关键词权重值
                                      allowPOS=('n','nr','ns')  # 仅包括指定词性的词
                                      )
    result = " ".join(keywords)
    return result

#========================================================
#  主题提取
#========================================================

def topic_LSI(doc, n):
    '''
    主题提取(LSI主题模型)
    INPUT  -> 输入文本, 主题数量
    OUTPUT -> 文本关键词
    '''
    wordlist = doc2words(doc)
    # 构建词典
    dictionary = corpora.Dictionary(documents=wordlist)
    # 语料词频,形成词袋
    corpus = [dictionary.doc2bow(sentence) for sentence in wordlist]
    # lsi模型，num_topics是主题的个数
    lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=n)
    # 打印所有主题，每个主题显示5个词
    for topic in lsi.print_topics(num_topics=n, num_words=8):
        print(topic)

def topic_LDA(doc, n):
    '''
    主题提取(LDA主题模型)
    INPUT  -> 输入文本, 主题数量
    OUTPUT -> 文本关键词
    '''
    wordlist = doc2words(doc)
    # 构建词典
    dictionary = corpora.Dictionary(documents=wordlist)
    # 语料词频,形成词袋
    corpus = [dictionary.doc2bow(sentence) for sentence in wordlist]
    # lda模型，num_topics是主题的个数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n)
    # 打印所有主题，每个主题显示5个词
    for topic in lda.print_topics(num_topics=n, num_words=8):
        print(topic)

