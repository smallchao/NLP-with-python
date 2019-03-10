#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np 
import os,re,math,sklearn,random,jieba,gensim,requests,json,difflib
from collections import Counter
from gensim import corpora, models, similarities
from gensim.test.utils import datapath
from gensim.models import word2vec, fasttext
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings; warnings.filterwarnings(action='ignore')

#========================================================
#  前置技术
#========================================================

def load_data(filename):
    '''
    读取文件
    '''
    file = os.path.splitext(filename)
    preffix, postfix = file
    postfix = postfix.lower()
    if postfix == '.csv':
        data = pd.read_csv(os.path.join(FILE_DIR, filename))
    elif postfix == '.txt':
        data = pd.read_table(os.path.join(FILE_DIR, filename))
    elif postfix == '.xls' or postfix == '.xlsx':
        data = pd.read_excel(os.path.join(FILE_DIR, filename))
    elif postfix == '.pkl':
        data = pd.read_pickle(os.path.join(FILE_DIR, filename))
    return data

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
#========================================================
#  断句
#========================================================

def doc2sentences(doc):
    '''
    文本断句功能(sentence segmentation)
    INPUT  -> 输入文本
    OUTPUT -> 句子列表
    '''
    cutLineFlag = '\?|\？|\!|\！|\。|\;|\；|\…|\【|\】'
    sentences = []
    doc = re.sub(r'\n+','。',doc)  # 换行改成句号（标题段无句号的情况） 
    doc = doc.replace('。。','。')
    doc = doc.replace('？。','。')
    doc = doc.replace('！。','。')  # 删除多余的句号
    sent_cut = re.split(cutLineFlag, str(doc))
    for sentence in sent_cut:
        if len(sentence) < 4: # 删除换行符、一个字符等
            continue
        sentence = sentence.strip('　 ')
        sentences.append(sentence)
    return sentences

#========================================================
#  分词
#========================================================

def tokenization(doc, userlist=False, stopwords=False):
    '''
    分词功能
    INPUT  -> 输入文本, 是否使用自定义词典, 是否过滤停用词
    OUTPUT -> 分词列表
    '''
    try:
        if userlist:
            jieba.load_userdict(FILE_DIR+"/userdict.txt")
        cleaner = re.compile(r'<[^>]+>',re.S)
        doc_clean = cleaner.sub('', doc)
        wordlist = jieba.lcut(doc_clean, cut_all=False)
        # 去数字
        wordlist = [v for v in wordlist if not str(v).isdigit()]
        # 去左右空格
        wordlist = list(filter(lambda x:x.strip(), wordlist))
        # 去掉长度小于1的词
        # wordlist = list(filter(lambda x:len(x)>1, wordlist))
        # 过滤标点符号(一般的中文文本分析都不需要标点)
        wordlist = list(filter(lambda x:x not in ['，',',','。','?','？','、',':','：','!','！','(',')','（','）','《','》','-','`','~','#','@','&','*','$','%','^','+','_','/','“','”','[',']','{','}'], wordlist)) 
        if stopwords:
            # stopwords = pd.read_table(FILE_DIR+"/stopwords.txt").values
            stopwords = set('的 和 会 具有 形成 可以 通过'.split())
            wordlist = list(filter(lambda x:x not in stopwords, wordlist))
        return wordlist
    except Exception:
        print(doc)

def doc2words(doc, sent_cut=True):
    '''
    断句+分词
    INPUT  -> 输入文本, 是否分句
    OUTPUT -> 分词列表
    '''
    wordlist = []
    if sent_cut:
        sentences = doc2sentences(doc)
        for sentence in sentences:
            wordlist.append(tokenization(sentence))
    else:
        for word in tokenization(doc):
            wordlist.append(word)
    return wordlist

#========================================================
#  文本向量化:稀疏编码，缺乏意义表达
#  包括one-hot向量化、tf向量化、tf-idf向量化、哈希向量化
#========================================================

def build_dictionary(wordlist):
    '''
    构建词典
    INPUT  -> 分词列表
    OUTPUT -> 词典
    '''
    # 求并集
    words = [x for item in wordlist for x in item]
    # 去重
    dictionary = sorted(set(words), key=words.index)
    return dictionary

def doc2onehot(doc, dictionary):
    '''
    one-hot编码
    INPUT  -> 输入文本, 词典
    OUTPUT -> one-hot编码
    '''
    wordlist = doc2words(doc)
    # V是编码维度
    V = len(dictionary)
    onehot = np.zeros(V)
    for line in wordlist:
        for word in line:
            if word in dictionary:
                pos = dictionary.index(word)
                onehot[pos] = 1
    return onehot

def doc2tf(doc, dictionary):
    '''
    tf编码/词频编码
    INPUT  -> 输入文本, 词典
    OUTPUT -> tf编码
    '''
    wordlist = doc2words(doc)
    # V是编码维度
    V = len(dictionary)
    tf = np.zeros(V)
    for line in wordlist:
        for word in line:
            if word in dictionary:
                pos = dictionary.index(word)
                tf[pos] += 1
    return tf

def doc2tf_sk(doc, dictionary):
    '''
    词频编码(基于sklearn)
    INPUT  -> 输入文本, 词典
    OUTPUT -> tf编码
    '''
    wordlist = tokenization(doc)
    wordlist = ' '.join(wordlist)  # 转换为空格分隔
    vectorizer = CountVectorizer(vocabulary=dictionary,
                                 token_pattern='(?u)\\b\\w+\\b', # 匹配字符,有至少一个文字类字符(A-Z、a-z、0-9以及下划线_),+代表至少
                                 # token_pattern='(?u)\\b\\w\\w+\\b', # 匹配字符,有至少两个文字类字符
                                 # lowercase=True,  # 转换为小写
                                 )
    tf = vectorizer.fit_transform([wordlist]).toarray()
    return tf[0]

def docs2tfidf_sk(docs):
    '''
    tf-idf编码(基于sklearn),适合文本聚类、内容比较
    一个词的权重由TF * IDF 表示，其中TF表示词频，即一个词在这篇文本中出现的频率；IDF表示逆文档频率，即一个词在所有文本中出现的频率倒数。
    因此，一个词在某文本中出现的越多，在其他文本中出现的越少，则这个词能很好地反映这篇文本的内容，权重就越大。
    INPUT  -> 文本集
    OUTPUT -> 文本集的tf-idf编码
    '''
    temp = []
    wordlist = []
    for doc in docs:
        temp.append(doc2words(doc, False))
        wordlist.append(' '.join(tokenization(doc)))
    dictionary = build_dictionary(temp)
    vectorizer = TfidfVectorizer(vocabulary=dictionary,
                                 token_pattern='(?u)\\b\\w+\\b', # 匹配字符,有至少一个文字类字符(A-Z、a-z、0-9以及下划线_),+代表至少
                                 # token_pattern='(?u)\\b\\w\\w+\\b', # 匹配字符,有至少两个文字类字符
                                 # lowercase=True,  # 转换为小写
                                 )
    tfidf = vectorizer.fit_transform(wordlist).toarray()
    print(tfidf)
    return tfidf

def doc2hashing_sk(doc):
    '''
    hashing编码(基于sklearn)
    哈希向量化可以缓解TfidfVectorizer在处理高维文本时内存消耗过大的问题
    这种方法不需要词汇表，可使用任意长度的向量来表示，但这种方法不可逆，不能再转化成对应的单词，不过很多监督学习任务并不care
    INPUT  -> 输入文本
    OUTPUT -> hashing编码
    '''
    wordlist = tokenization(doc)
    wordlist = ' '.join(wordlist)  # 转换为空格分隔
    vectorizer = HashingVectorizer(n_features=10,  # 向量长度 
                                   token_pattern='(?u)\\b\\w+\\b', # 匹配字符,有至少一个文字类字符(A-Z、a-z、0-9以及下划线_),+代表至少
                                   # token_pattern='(?u)\\b\\w\\w+\\b', # 匹配字符,有至少两个文字类字符
                                   # lowercase=True,  # 转换为小写
                                   )
    hashing = vectorizer.fit_transform([wordlist]).toarray()
    return hashing[0]

#========================================================
#  文本向量化:稠密编码
#  词嵌入向量(CBOW模型和Skip-gram模型)
#========================================================

#========================================================
#  文本向量化:稠密编码
#  gensim词嵌入、fastText词嵌入
#========================================================

def make_corpus(path, filename):
    '''
    对原始语料预处理,生成可供训练的语料
    INPUT  -> 文件目录, 文件名
    '''
    origin_file = path+'/'+filename
    train_file = path+'/'+filename+'_cut.txt'
    try:
        fi = open(origin_file, 'rb')
    except BaseException as e:
        print(Exception, ":", e)
    doc = fi.read()
    new_doc = tokenization(doc)
    str_out = ' '.join(new_doc)
    fo = open(train_file, 'w', encoding='utf-8')
    fo.write(str_out)
    fo.close()

def word2vec_train(train_file, save_model_name):
    '''
    训练词向量模型(word2vec版)
    INPUT  -> 训练语料地址, 模型保存的名称
    '''
    corpus_path = FILE_DIR+'/'+train_file
    model_path = FILE_DIR+'/'+save_model_name+'.bin'

    corpus_file = word2vec.Text8Corpus(corpus_path)
    # 训练skip-gram模型
    model = word2vec.Word2Vec(corpus_file,  # 训练语料
                              sg=1,  # 1是skip-gram算法(对低频词敏感),0是CBOW算法
                              size=150,  # 是输出词向量的维数,一般取100-200间(太小会导致映射冲突,太大消耗内存)
                              window=5,  # 句子中当前词语目标词之间的最大距离(前看n个词,后看n个词)
                              min_count=1, # 对词进行过滤,小于n的词会被忽视,默认为5
                              alpha=0.025,  # 学习率
                              workers=4,  # 并发训练时候的线程数，仅当Cython安装的情况下才会起作用
                              iter=5,  # 训练周期,默认是5
                              )
    # model.save(FILE_DIR+'/'+save_model_name)
    # 以二进制类型保存模型以便重用
    model.wv.save_word2vec_format(model_path, binary=True)

def fastText_train(train_file, save_model_name):
    '''
    训练词向量模型(fastText版)
    INPUT  -> 训练语料地址, 模型保存的名称
    '''
    corpus_path = FILE_DIR+'/'+train_file
    model_path = FILE_DIR+'/'+save_model_name+'.bin'

    corpus_file = datapath(corpus_path)
    model = fasttext.FastText(corpus_file, # 训练语料
                              sg=1,  # 1是skip-gram算法(对低频词敏感),0是CBOW算法
                              size=150,  # 是输出词向量的维数,一般取100-200间(太小会导致映射冲突,太大消耗内存) 
                              window=5,  # 句子中当前词语目标词之间的最大距离(前看n个词,后看n个词) 
                              min_count=1, # 对词进行过滤,小于n的词会被忽视,默认为5
                              alpha=0.025,  # 学习率
                              workers=4,  # 并发训练时候的线程数，仅当Cython安装的情况下才会起作用
                              iter=5,  # 训练周期,默认是5
                              )

    # model.save(FILE_DIR+'/'+save_model_name)
    # 以二进制类型保存模型以便重用
    model.wv.save_word2vec_format(model_path, binary=True)

def doc2vec(doc, save_model_name):
    '''
    基于词嵌入的文本向量化编码
    INPUT  -> 输入文本, 训练好的模型
    OUTPUT -> 词向量编码
    '''
    wordlist = doc2words(doc, False)
    model = gensim.models.KeyedVectors.load_word2vec_format(FILE_DIR+'/'+save_model_name+'.bin', binary=True)

    doc_vecs = []
    for word in wordlist:
        doc_vecs.append(model[word].reshape((1, 150)))
    result = np.array(np.concatenate(doc_vecs), dtype='float')

    return result

def wordvec_sim(word1, word2, save_model_name):
    '''
    比较两词相似度
    INPUT  -> 词1, 词2, 训练好的模型
    OUTPUT  -> 相似度
    '''
    # model = gensim.models.Word2Vec.load(FILE_DIR+'/'+save_model_name)
    model = gensim.models.KeyedVectors.load_word2vec_format(FILE_DIR+'/'+save_model_name+'.bin', binary=True)

    return model.similarity(word1, word2)
    
def wordvec_most_sim(word, save_model_name):
    '''
    返回最相似的词
    INPUT  -> 词, 训练好的模型
    '''
    # model = gensim.models.Word2Vec.load(FILE_DIR+'/'+save_model_name)
    model = gensim.models.KeyedVectors.load_word2vec_format(FILE_DIR+'/'+save_model_name+'.bin', binary=True)

    print("-------------------------------\n")
    word_list = model.most_similar(word, topn=10)
    print(u"和"+word+u"最相关的词有：\n")
    for item in word_list:
        print(item[0], item[1])  # 打印相关词和相关性
    print("-------------------------------\n")

#========================================================
#  文本比较
#========================================================

def readline(filename):
    '''
    行读取
    INPUT  -> 文件名
    OUTPUT -> 行读取结果
    '''
    try:
        with open(filename, 'r') as f:
            return f.readlines()
    except IOError:
        print("ERROR: 没有找到文件:%s或读取文件失败！" % filename)

def compare_doc(file1, file2):
    '''
    文档比较
    INPUT  -> 文件1, 文件2
    OUTPUT -> 差分结果
    '''
    doc1 = readline(file1)
    doc2 = readline(file2)
    d = difflib.HtmlDiff()
    result = d.make_file(doc1, doc2)
    print(result)
    with open('result.html', 'w', encoding='utf-8') as f:
        f.writelines(result)

def compare_tdidf(doc1, doc2):
    '''
    基于tfidf的文档相似度
    INPUT  -> 文本1, 文本2
    OUTPUT -> 相似度
    '''
    docs = []
    docs.append(doc1)
    docs.append(doc2)
    tfidf = docs2tfidf_sk(docs)
    return cosin_sim(tfidf[0], tfidf[1])
