# 配置参数

class TrainingConfig(object):
    # 训练参数
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001

class ModelConfig(object):
    # 模型参数
    embeddingSize = 200
    numFilters = 128  #每个尺寸的卷积核的个数都为128
    filterSizes = [2, 3, 4, 5]  # 我们在论文的基础上加入了size=2的卷积核，卷积层只有一层
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0  # L2正则化系数

class Config(object):
    sequenceLength = 200  # 取了所有序列长度的均值
    batchSize = 140

    dataSource = "./data/preProcess/labeledTrain.csv"
    testSource = "./data/rawData/testData.tsv"

    stopWordSource = "./data/english"  # 停用词表

    numClasses = 2

    rate = 0.8  # 训练集的比例

    training = TrainingConfig()

    model = ModelConfig()
