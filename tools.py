#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np 
import networkx as nx
import os,re,math,sklearn,random,jieba,gensim,requests,json,folium,webbrowser,difflib
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
# 设置matplotlib正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False 
from collections import Counter
from gensim import corpora, models, similarities
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB  # 参数存在负值用GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
from wordcloud import WordCloud
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
#  文本可视化:基于文本内容
#  词云、分布图 和 Document Cards
#========================================================
def image_wordcloud(text, pil_im):
    '''
    生成词云图
    INPUT  -> 文本, 图像文件
    '''
    mask_image_arr = np.array(pil_im, 'f')
    # 通过jieba分词进行分词并通过空格分隔
    wordlist_after_jieba = jieba.cut(text, cut_all = False)
    wordlist_space_split = " ".join(wordlist_after_jieba)
    # 生成词云
    cloud = WordCloud(font_path='C:/Users/Windows/fonts/simkai.ttf',  # 英文的不导入也并不影响，若是中文的或者其他字符需要选择合适的字体包
                      background_color='white',  # 设置背景颜色
                      mask=mask_image_arr,    # 设置掩膜,产生词云背景的区域
                      max_words=2000,    # 设置最大显示的字数
                      max_font_size=80,  # 设置字体最大值
                      random_state=40,    # 设置有多少种配色方案
                      margin=5).generate(wordlist_space_split)
    # 保存图片
    cloud.to_file(os.path.join(FILE_DIR, 'my_wordcloud.png'))


#========================================================
#  文本可视化:基于文本关系
#  树状图、节点连接的网络图、力导向图、叠式图和 Word Tree
#========================================================

def plot_network(table):
    '''
    绘制网络关系图
    INPUT  -> 文本, 图像文件
    '''
    # 所有节点(list形式)
    source = table['source'].values.tolist()
    target = table['target'].values.tolist()
    nodes = list(set(source + source))
    # 所有的边(list形式,元素是成对的节点)
    edges = [(table.loc[index,'source'], table.loc[index,'target']) for index in table.index]   
    edges =  list(set(edges))

    colors = ['red', 'green', 'blue', 'yellow']
    
    G = nx.Graph()  # 分析图
    # G = nx.DiGraph()  # 有向图
    # G = nx.Multigraphs()  # 多边图
    # G = nx.MultiDiGraph()  # 多边有向图
    # G = nx.path_graph()  # 线形图
    # G = nx.cycle_graph()  # 环形图
    # G = nx.cubical_graph()  # 立方体图
    # G = nx.petersen_graph()  # 彼得森图

    # 添加节点列表
    G.add_nodes_from(nodes)
    # 添加边列表
    G.add_edges_from(edges)

    pos = nx.random_layout(G)  # 节点位置为随机分布
    # pos = nx.circular_layout(G)  # 节点位置为环形分布
    # pos = nx.spectral_layout(G)  # 节点位置为谱分布
    
    # 作图,设置节点,边,标签
    nx.draw_networkx_nodes(G, pos, alpha=0.2, node_size=1200, node_color=colors)
    nx.draw_networkx_edges(G, pos, node_color='r', alpha=0.3, style='dashed')
    nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=0.5, font_size=5)

    # nx.draw_circular(G)
    # nx.draw(G, with_labels=True, node_size=1000, node_color = colors)
    
    plt.show() 

#========================================================
#  文本可视化:基于多层面信息
#  地理热力图、ThemeRiver、SparkClouds、TextFlow 和基于矩阵视图的情感分析可视化
#========================================================

def getlonlat_by_address(address):
    '''
    地理编码:将地理名词通过高德地图转换为经纬度
    INPUT  -> 地理名词
    OUTPUT -> 高德坐标
    '''
    url = 'https://restapi.amap.com/v3/geocode/geo'
    key = '5bdc284755f32a8f48a5c179507a551d'
    uri = url+'?'+'address='+address+'&output=json&key='+key
    res = requests.get(uri).text
    result = json.loads(res)['geocodes'][0]['location']
    return result

def plot_address(table):
    '''
    绘制地图热力图
    INPUT  -> 地理数据文件
    '''
    lon = np.array(table['lon'])
    lat = np.array(table['lat'])
    pop = np.array(table['count'], dtype=float)
    # 将数据制作成[lats,lons,weights]的形式
    data_box = [[lon[i],lat[i],pop[i]] for i in range(len(table['lon']))]
    # 绘制map，初始缩放比例5倍
    map_osm = folium.Map(location=[35, 110], zoom_start=5)
    # 将热力图加到前面建立的地图中
    HeatMap(data_box).add_to(map_osm)
    map_osm.save(os.path.join(FILE_DIR, 'hot.html'))
    webbrowser.open(os.path.join(FILE_DIR, 'hot.html'))

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

def wordvec_corpus(path, filename):
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

def wordvec_train(train_file, save_model_name):
    '''
    训练词向量模型
    INPUT  -> 训练语料地址, 模型保存的名称
    '''
    corpus_path = FILE_DIR+'/'+train_file
    model_path = FILE_DIR+'/'+save_model_name+'.bin'

    tokenized = word2vec.Text8Corpus(corpus_path)
    # 训练skip-gram模型
    model = gensim.models.Word2Vec(tokenized,  # 训练语料
                                   sg=1,  # 1是skip-gram算法(对低频词敏感),0是CBOW算法
                                   size=150,  # 是输出词向量的维数,一般取100-200间(太小会导致映射冲突,太大消耗内存)
                                   window=5,  # 句子中当前词语目标词之间的最大距离(前看n个词,后看n个词)
                                   min_count=1, # 对词进行过滤,小于n的词会被忽视,默认为5
                                   workers=4,  # 并发训练时候的线程数，仅当Cython安装的情况下才会起作用
                                   )
    # model.save(FILE_DIR+'/'+save_model_name)
    # 以二进制类型保存模型以便重用
    model.wv.save_word2vec_format(model_path, binary=True)

def doc2vec_wv(doc, save_model_name):
    '''
    词向量编码(word2vec版)
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

#========================================================
#  文本摘要
#========================================================

class Summary():
    def cutSentence(self, doc):
        '''
        正文断句
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

    def getKeywords(self, title, sentences, n=10):
        '''
        提取标题和正文中的特征词
        '''
        wordlist = []
        for sentence in sentences:
            split_result = tokenization(sentence)
            for i in split_result:
                wordlist.append(i)
        # 统计词频TF
        tf = Counter(wordlist)
        # 正文中提取特征
        words_body = [i[0] for i in tf.most_common(n) if i[1]>1] # 在topN中排除出现次数少于2次的词
        # 标题中提取特征
        words_title = [word for word in tokenization(title)]
        # 正文关键词与标题关键词取并集,作为最终关键词
        Keywords = list(set(words_body)|set(words_title)) 
        print (' '.join(Keywords))
        return Keywords

    def getTopNSentences(self, sentences, keywords, n=3):
        '''
        提取topN句子
        '''
        sents_score = {}
        len_sentences = len(sentences)
        # 初始化句子重要性得分，并计算句子平均长度
        len_avg = 0
        len_min = len(sentences[0])
        len_max = len(sentences[0])
        for sent in sentences:
            sents_score[sent] = 0
            l = len(sent)
            len_avg += l
            if len_min > l:
                len_min = l
            if len_max < l:
                len_max = l
        len_avg = len_avg / len_sentences
        # 计算句子权重得分
        for sent in sentences:
            l = len(sent)
            if l < (len_min + len_avg) / 2 or l > (3 * len_max - 2 * len_avg) / 4:  # 不考虑句长在指定范围外的句子
                continue
            words = []
            sent_words = jieba.cut(sent)
            for i in sent_words:
                words.append(i)
            keywords_cnt = 0
            len_sent = len(words)
            if len_sent == 0:
                continue
            for word in words:
                if word in keywords:
                    keywords_cnt += 1
            score = keywords_cnt * keywords_cnt * 1.0 / len_sent
            sents_score[sent] = score
            if sentences.index(sent) == 0:  # 提高首句权重
                sents_score[sent] = 2 * score
        # 排序 
        dict_list = sorted(sents_score.items(),key=lambda x:x[1],reverse=True)
        # 返回topN
        sents_topN = []
        for i in dict_list[:n]:
            sents_topN.append(i[0])
        sents_topN = list(set(sents_topN))
        # 按比例提取
        if len_sentences <= 5:
            sents_topN = sents_topN[:1]
        elif len_sentences < 9:
            sents_topN = sents_topN[:2]
        return sents_topN

    def sents_sort(self, sents_topN, sentences):
        '''
        恢复topN句子在文中的相对顺序
        '''
        keysents = []
        for sent in sentences:
            if sent in sents_topN and sent not in keysents:
                keysents.append(sent)
        keysents = self.post_processing(keysents)

        return keysents

    def post_processing(self, keysents):
        '''
        输出摘要前的最终处理
        '''
        # 删除不完整句子中的详细部分
        detail_tags = ['，一是','：一是','，第一，','：第一，','，首先，','；首先，']
        for i in keysents:
            for tag in detail_tags:
                index = i.find(tag)
                if index != -1:
                    keysents[keysents.index(i)] = i[:index]
        # 删除编号
        for i in keysents:
            regex = re.compile(r'^一、|^二、|^三、|^三、|^四、|^五、|^六、|^七、|^八、|^九、|^十、|^\d{1,2}、|^\d{1,2} ')
            result = re.findall(regex,i)
            if result:
                keysents[keysents.index(i)] = re.sub(regex,'',i)
        # 删除备注性质的句子
        for i in keysents:
            regex = re.compile(r'^注\d*：')
            result = re.findall(regex,i)
            if result:
                keysents.remove(i)
        # 删除句首括号中的内容
        for i in keysents:
            regex = re.compile(r'^.∗')
            result = re.findall(regex,i)
            if result:
                keysents[keysents.index(i)] = re.sub(regex,'',i)
        # 删除来源(空格前的部分)
        for i in keysents:
            regex = re.compile(r'^.{1,20} ')
            result = re.findall(regex,i)
            if result:
                keysents[keysents.index(i)] = re.sub(regex,'',i)
        # 删除引号部分(如：银行间债市小幅下跌，见下图：)
        for i in keysents:
            regex = re.compile(r'，[^，]+：$')
            result = re.findall(regex,i)
            if result:
                keysents[keysents.index(i)] = re.sub(regex,'',i)
        return keysents

    def main(self,title, doc):
        sentences = self.cutSentence(doc)
        keywords = self.getKeywords(title, sentences, n=8)   # 标题对于找到文章主题很重要
        sents_topN = self.getTopNSentences(sentences, keywords, n=3)
        keysents = self.sents_sort(sents_topN, sentences)
        print(keysents)
        return keysents

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
