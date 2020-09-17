
import jieba,sys,os,numpy,math,re
from gensim import corpora, models, similarities
from collections import defaultdict

# 定义文件路径
f1 = sys.argv[1]
f2 = sys.argv[2]
##f3 = sys.argv[3]
##f4 = sys.argv[4]
##f5 = sys.argv[5]
##f6 = sys.argv[6]
##f7 = sys.argv[7]
##f8 = sys.argv[8]
##f9 = sys.argv[9]


##f0 = "D:\sim_0.8\orig.txt" (对比文件
f0 = sys.argv[3]
##bushi10
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


file0 = open(f0, encoding='utf-8').read()


# 读取文件内容
s1 = open(f1, encoding='utf-8').read()
s2 = open(f2, encoding='utf-8').read()
##s3 = open(f3,encoding="utf-8").read()
##s4 = open(f4,encoding="utf-8").read()
##s5 = open(f5,encoding="utf-8").read()
##s6 = open(f6,encoding="utf-8").read()
##s7 = open(f7,encoding="utf-8").read()
##s8 = open(f8,encoding="utf-8").read()
##s9 = open(f9,encoding="utf-8").read()

#利用jieba分词与停用词表，将词分好并保存到向量中
stopwords=[]
fstop=open('D:/results/stopwords_file.txt','r',encoding='utf-8-sig')
for eachWord in fstop:
    eachWord = re.sub("\n", "", eachWord)
    stopwords.append(eachWord)
fstop.close()
s1_cut = [i for i in jieba.cut(s1, cut_all=True) if (i not in stopwords) and i!='']
s2_cut = [i for i in jieba.cut(s2, cut_all=True) if (i not in stopwords) and i!='']
##s3_cut = [i for i in jieba.cut(s3, cut_all=True) if (i not in stopwords) and i!='']
##s4_cut = [i for i in jieba.cut(s4, cut_all=True) if (i not in stopwords) and i!='']
##s5_cut = [i for i in jieba.cut(s5, cut_all=True) if (i not in stopwords) and i!='']
##s6_cut = [i for i in jieba.cut(s6, cut_all=True) if (i not in stopwords) and i!='']
##s7_cut = [i for i in jieba.cut(s7, cut_all=True) if (i not in stopwords) and i!='']
##s8_cut = [i for i in jieba.cut(s8, cut_all=True) if (i not in stopwords) and i!='']
##s9_cut = [i for i in jieba.cut(s9, cut_all=True) if (i not in stopwords) and i!='']
word_set = set(s1_cut).union(set(s2_cut))
##word_set = set(word_set).union(set(s3_cut))
##word_set = set(word_set).union(set(s4_cut))
##word_set = set(word_set).union(set(s5_cut))
##word_set = set(word_set).union(set(s6_cut))
##word_set = set(word_set).union(set(s7_cut))
##word_set = set(word_set).union(set(s8_cut))
##word_set = set(word_set).union(set(s9_cut))


#用字典保存两篇文章中出现的所有词并编上号
word_dict = dict()
i = 0
for word in word_set:
    word_dict[word] = i
    i += 1


#根据词袋模型统计词在每篇文档中出现的次数，形成向量
s1_cut_code = [0]*len(word_dict)

for word in s1_cut:
    s1_cut_code[word_dict[word]]+=1

s2_cut_code = [0]*len(word_dict)
for word in s2_cut:
    s2_cut_code[word_dict[word]]+=1
##
##s3_cut_code = [0]*len(word_dict)
##
##for word in s1_cut:
##    s3_cut_code[word_dict[word]]+=1
##
##s4_cut_code = [0]*len(word_dict)
##for word in s2_cut:
##    s4_cut_code[word_dict[word]]+=1
##
##s5_cut_code = [0]*len(word_dict)
##
##for word in s1_cut:
##    s5_cut_code[word_dict[word]]+=1
##
##s6_cut_code = [0]*len(word_dict)
##for word in s2_cut:
##    s6_cut_code[word_dict[word]]+=1
##
##s7_cut_code = [0]*len(word_dict)
##
##for word in s1_cut:
##    s7_cut_code[word_dict[word]]+=1
##
##s8_cut_code = [0]*len(word_dict)
##for word in s2_cut:
##    s8_cut_code[word_dict[word]]+=1
##
##s9_cut_code = [0]*len(word_dict)
##
##for word in s1_cut:
##    s9_cut_code[word_dict[word]]+=1


# 计算余弦相似度
sum = 0
sq1 = 0
sq2 = 0
##sq3 = 0
##sq4 = 0
##sq5 = 0
##sq6 = 0
##sq7 = 0
##sq8 = 0
##sq9 = 0
for i in range(len(s1_cut_code)):
    sum += s1_cut_code[i] * s2_cut_code[i]
    sq1 += pow(s1_cut_code[i], 2)
    sq2 += pow(s2_cut_code[i], 2)

try:
    result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 3)
except ZeroDivisionError:
    result = 0.0

result=result*100
print("\n余弦相似度为：%.2f%%"%result)
f100 = sys.argv[4]

##file = open(f100,'w', encoding='UTF-8')
##result = (str)result
##file.write(result)
##file.close()
##print("文本相似度为：%.2f%%"%result,file=file0)
