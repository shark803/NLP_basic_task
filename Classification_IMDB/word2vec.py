import pandas as pd
from bs4 import BeautifulSoup

with open("./data/rawData/unlabeledTrainData.tsv", "r") as f:
    unlabeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]
    
with open("./data/rawData/labeledTrainData.tsv", "r") as f:
    labeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 3]

unlabel = pd.DataFrame(unlabeledTrain[1: ], columns=unlabeledTrain[0])
label = pd.DataFrame(labeledTrain[1: ], columns=labeledTrain[0])

def cleanReview(subject):
	# 数据处理函数
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '')
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)
    
    return newSubject
    
unlabel["review"] = unlabel["review"].apply(cleanReview)
label["review"] = label["review"].apply(cleanReview)

# 将有标签的数据和无标签的数据合并
newDf = pd.concat([unlabel["review"], label["review"]], axis=0) 
# 保存成txt文件
newDf.to_csv("./wordEmbdiing.txt", index=False)

import logging
import gensim
from gensim.models import word2vec

# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。
sentences = word2vec.LineSentence("./wordEmbdiing.txt")

# 训练模型，词向量的长度设置为200， 迭代次数为8，采用skip-gram模型，模型保存为bin格式
model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)  
model.wv.save_word2vec_format("./word2Vec" + ".bin", binary=True) 

# 加载bin格式的模型
#wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin", binary=True)


