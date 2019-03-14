#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np 
import networkx as nx
import os,re,math,jieba,requests,json,folium,webbrowser
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
# 设置matplotlib正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False 
from wordcloud import WordCloud
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

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

def plot_network(table, direct, weight):
    '''
    绘制网络关系图
    INPUT  -> 表格, 连接形式, 是否有权重
    '''
    colors = ['red', 'green', 'blue', 'yellow']
    
    # 从pandas快速构建图
	# G = nx.from_pandas_edgelist(routes_us, source = 'source', target = 'target', edge_attr = 'numbers', create_using = nx.DiGraph())
	
    # 传统构建图
    if direct == 'directed':
        G = nx.DiGraph()  # 无多重边有向图
    elif direct == 'undirected':
        G = nx.Graph()   # 无多重边无向图
    elif direct == 'multi.directed':
        G = nx.MultiDiGraph()  # 有多重边有向图
    elif direct == 'multi.undirected':
        G = nx.MultiGraph()  # 有多重边无向图

    # 加载数据(方法一)
    if weight:
        edge_list = [tuple(g_list.values()) for g_list in table.edges[['source','target','weight']]]
        G.add_weighted_edges_from(edge_list)
    else:
        edge_list = [tuple(g_list.values()) for g_list in table.edges[['source','target']]]
        G.add_edges_from(edge_list)
        # 所有节点(list形式)
        # source = table['source'].values.tolist()
        # target = table['target'].values.tolist()
        # nodes = list(set(source + source))
        # 所有的边(list形式,元素是成对的节点)
        # edges = [(table.loc[index,'source'], table.loc[index,'target']) for index in table.index]   
        # edges =  list(set(edges))
        # 添加节点列表
        # G.add_nodes_from(nodes)
        # 添加边列表
        # G.add_edges_from(edges)

    pos = nx.random_layout(G)  # 节点位置为随机分布
    # pos = nx.circular_layout(G)  # 节点位置为环形分布
    # pos = nx.spectral_layout(G)  # 节点位置为谱分布
	
	# 作图(方法一)
    # nx.draw(G, pos = nx.random_layout(G), with_labels=True, font_size =18, node_size=1000, node_color = colors, edge_color = 'r')
	
	# 作图(方法二)：分别设置节点,边,标签
    nx.draw_networkx_nodes(G, pos, alpha=0.2, node_size=1200, node_color=colors)
    nx.draw_networkx_edges(G, pos, node_color='r', alpha=0.3, style='dashed')
    nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=0.5, font_size=5)
    
	# plt.savefig("network.png")
	# nx.write_gexf(G, 'network.gexf')  # gexf格式文件可以导入gephi中进行分析
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
