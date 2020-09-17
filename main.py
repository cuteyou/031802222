# -*- coding: utf-8 -*-

import jieba,sys,os,numpy
from gensim import corpora, models, similarities
from collections import defaultdict

# 定义文件路径

f1 = sys.argv[1]
f2 = sys.argv[2]
f3 = sys.argv[3]
f4 = sys.argv[4]
f5 = sys.argv[5]
f6 = sys.argv[6]
f7 = sys.argv[7]
f8 = sys.argv[8]
f9 = sys.argv[9]
##cmd： C:\Users\林逸丽\AppData\Local\Programs\Python\Python37>python main2.py D:\sim_0.8\orig_0.8_add.txt D:\sim_0.8\orig_0.8_del.txt D:\sim_0.8\orig_0.8_dis_1.txt D:\sim_0.8\orig_0.8_dis_3.txt D:\sim_0.8\orig_0.8_dis_7.txt D:\sim_0.8\orig_0.8_dis_10.txt D:\sim_0.8\orig_0.8_dis_15.txt D:\sim_0.8\orig_0.8_mix.txt D:\sim_0.8\orig_0.8_rep.txt D:\sim_0.8\orig.txt
##否则，f1 = sys.argv[1]
##IndexError: list index out of range


##f0 = "D:\sim_0.8\orig.txt" (对比文件
##f1 = "D:\sim_0.8\orig_0.8_del.txt"
##f2 = "D:\sim_0.8\orig_0.8_add.txt"
##f1 = "D:\sim_0.8\orig_0.8_add.txt"
##f2 = "D:\sim_0.8\orig_0.8_del.txt"
##f3 = "D:\sim_0.8\orig_0.8_dis_1.txt"
##f4 = "D:\sim_0.8\orig_0.8_dis_3.txt"
##f5 = "D:\sim_0.8\orig_0.8_dis_7.txt"
##f6 = "D:\sim_0.8\orig_0.8_dis_10.txt"
##f7 = "D:\sim_0.8\orig_0.8_dis_15.txt"
##f8 = "D:\sim_0.8\orig_0.8_mix.txt"
##f9 = "D:\sim_0.8\orig_0.8_rep.txt"



# 读取文件内容
c1 = open(f1, encoding='utf-8').read()
c2 = open(f2, encoding='utf-8').read()
c3 = open(f3,encoding="utf-8").read()
c4 = open(f4,encoding="utf-8").read()
c5 = open(f5,encoding="utf-8").read()
c6 = open(f6,encoding="utf-8").read()
c7 = open(f7,encoding="utf-8").read()
c8 = open(f8,encoding="utf-8").read()
c9 = open(f9,encoding="utf-8").read()
# jieba 进行分词
data1 = jieba.cut(c1)
data2 = jieba.cut(c2)
data3 = jieba.cut(c3)
data4 = jieba.cut(c4)
data5 = jieba.cut(c5)
data6 = jieba.cut(c6)
data7 = jieba.cut(c7)
data8 = jieba.cut(c8)
data9 = jieba.cut(c9)

data11 = ""
# 获取分词内容
for i in data1:
    data11 += i + " "
    
data21 = ""
# 获取分词内容
for i in data2:
    data21 += i + " "

data31 = ""
# 获取分词内容
for i in data3:
    data31 += i + " "

data41 = ""
# 获取分词内容
for i in data4:
    data41 += i + " "

data51 = ""
# 获取分词内容
for i in data5:
    data51 += i + " "

data61 = ""
# 获取分词内容
for i in data6:
    data61 += i + " "

data71 = ""
# 获取分词内容
for i in data7:
    data71 += i + " "

data81 = ""
# 获取分词内容
for i in data8:
    data81 += i + " "

data91 = ""
# 获取分词内容
for i in data9:
    data91 += i + " "

    

doc1 = [data11, data21,data31,data41,data51,data61,data71,data81,data91]
# print(doc1)

t1 = [[word for word in doc.split()]
      for doc in doc1]
# print(t1)

# # frequence频率
freq = defaultdict(int)
for i in t1:
    for j in i:
        freq[j] += 1
# print(freq)

# 限制词频
t2 = [[token for token in k if freq[j] >= 3]
      for k in t1]
##print(t2)

# corpora语料库建立字典
dic1 = corpora.Dictionary(t2)
dic1.save("D:/results/aaa.txt")

# 对比文件
##f0 = "D:\sim_0.8\orig.txt"
f0 = sys.argv[10]

c0 = open(f0, encoding='utf-8').read()
# jieba 进行分词
data0 = jieba.cut(c0)
data01 = ""
for i in data0:
    data01 += i + " "
new_doc = data01
##print(new_doc)

# doc2bow把文件变成一个稀疏向量
new_vec = dic1.doc2bow(new_doc.split())
# 对字典进行doc2bow处理，得到新语料库
new_corpor = [dic1.doc2bow(t3) for t3 in t2]
tfidf = models.TfidfModel(new_corpor)

# 特征数
featurenum = len(dic1.token2id.keys())

# similarities 相似之处
# SparseMatrixSimilarity 稀疏矩阵相似度
idx = similarities.SparseMatrixSimilarity(tfidf[new_corpor], num_features=featurenum)
sims = idx[tfidf[new_vec]]
for i in sims:
    i=1-i
    print("%.2f"%i)
##f100=D:/results/ans.txt
    f100 = sys.argv[11]
numpy.savetxt(f100,sims)
