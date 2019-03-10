#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np 
import os,re,math,sklearn,random,jieba
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings; warnings.filterwarnings(action='ignore')
from tools import tokenization

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
