# 配置参数

class TrainingConfig(object):
    epoches = 2
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001
    
class ModelConfig(object):
    embeddingSize = 200
    
    hiddenSizes = [128]  # LSTM结构的神经元个数
    
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0
    
class Config(object):
    sequenceLength = 200  # 取了所有序列长度的均值
    batchSize = 128
    
    dataSource = "./data/preProcess/labeledTrain.csv"
    testSource = "./data/rawData/testData.tsv"

    stopWordSource = "./data/english"  # 停用词表
    
    numClasses = 2
    
    rate = 0.8  # 训练集的比例
    
    training = TrainingConfig()
    
    model = ModelConfig()

    

