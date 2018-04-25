import re
import operator
import feedparser

from numpy import *

# 创建实验样本
def loadDataSet():
    posttingList = [['my', 'dog','has','flea','problems','help','please'],
                    ['maybe','not','take','help','to','dog','park','stupid'],
                    ['my','dalmatian','is','so','cute','I','love','him'],
                    ['stop','posting','stupid','worthless','garbage'],
                    ['mr','licks','ate','my','steak','how','to','stop','him'],
                    ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return posttingList, classVec

# 构建一个包含所有词且没有重复的词汇表
def createVocabList(dataSet):
    vocaSet = set([])
    for document in dataSet:
        # 并集构建过程随机，所以返回顺序也随机
        vocaSet = vocaSet | set (document)
    return list(vocaSet)

# 获取文档向量
def setOfWords2Vec(vocabList, inputSet):
    # 创建长度为len(vocabList)的 0 list
    returnVec = [0] * len(vocabList)
    # 逐个判断inputSet中的词汇是否存在于vocabList中，若存在将vocabList对应位置的returnVec标记为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('这个词：%s，不存在于我的词汇表中'% word)
    return returnVec

# 训练函数。
# 参数：文档矩阵、文档类别标签
def trainNB0(trainMatrix, trainCategory):
    # 获取文档矩阵中所含文档数量
    numTrainDocs = len(trainMatrix)
    # 获取第一项文档矩阵包含的项目数（实际上就是词汇矩阵表的项目数）
    numWords = len(trainMatrix[0])
    # 文档类别标签求和/文档总数=文档为1的概率
    pAbusive = sum(trainCategory)/numTrainDocs
    # 创建长度等于词汇表项目数的两个1矩阵，对应1，0两种分类
    # 使用1矩阵是因为，若使用0矩阵，当某个词没有出现时，其概率为0，在之后的概率相乘时会使得结果总为0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 创建总词数计数变量，对应0，1两种类型
    # 由于上方的词汇矩阵使用了1为基数，故总词数计数变量也变为从2开始
    # 原本为1
    p0Denom = 2
    p1Denom = 2
    # 遍历每篇文章的向量
    for i in range(numTrainDocs):
        # 如果这篇文章的标签为1
        if trainCategory[i] == 1:
            # 词汇矩阵累加
            p1Num += trainMatrix[i]
            # 各词汇出现总次数累加
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 用各词汇出现概率/词汇出现总次数，得出每个词汇在1、0型文档中出现的频率
    # 这里取对数是因为，大部分词的出现概率极低，若直接使用概率值会导致在之后的概率相乘时出现下溢
    # 取对数不影响取到极值
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClasss1):
    # 将测试集的文档向量和训练模型逐项相乘，再逐项求和，并加上基值，得出测试集的偏向1或0的倾向
    p1 = sum(vec2Classify * p1Vec) + log(pClasss1)
    p0 = sum(vec2Classify * p0Vec) + log(1-pClasss1)
    # 比较倾向后返回结果
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    # 装载数据
    listOPosts, listClasses = loadDataSet()
    # 构建词汇表
    myVocabList = createVocabList(listOPosts)
    # 构建文档向量
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 进行训练
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    # 构建测试集
    testEntry = ['love', 'my', 'dalmatian']
    # 获取测试集的文档向量
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, '分类为：', classifyNB(thisDoc, p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, '分类为：', classifyNB(thisDoc, p0V, p1V, pAb))

# 词袋模型函数，当一个词多次出现时，增加其在文章矩阵中的计数
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 词汇切割函数
def textParse(bigString):
    # 分割单词，并利用正则表达式去掉符号
    listOfTokens = re.split('\\W*',bigString)
    # 将长度大于2的单词转换为全小写后组成list返回
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    # 文档list
    docList = []
    # 分类标签list
    classList = []
    # 全list（文档list变量会被冲掉，全list同于储存所有出现过的文档）
    fullText = []
    for i in range(1,26):
        # 逐个读取广告邮件文本
        wordList = textParse(open('./data/ch04/email/spam/%d.txt'% i).read())
        # 将读取到的广告文本添加到文档list中
        docList.append(wordList)
        # 将读取到的广告文本添加到全list中
        fullText.extend(wordList)
        # 对分类标签list标记为1（广告文本）
        classList.append(1)
        # 随书附带的文件的第23个中的第一个?，编码不是GBK也不是UTF-8，建议手动修改
        # 读取非广告邮件文本
        wordList = textParse(open('./data/ch04/email/ham/%d.txt'% i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建词汇列表
    vocablist = createVocabList(docList)
    # 书中源代码为trainingSet = range(50)
    # 生成训练集编号
    # 由于广告邮件和正常邮件各25封，所以这里取50
    trainingSet = list(range(50))
    # 测试集
    testSet = []
    for i in range(10):
        # 随机选取十篇邮件
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 加入到测试集
        testSet.append(trainingSet[randIndex])
        # 同时删除对应的训练集编号
        trainingSet.pop(randIndex)
    # 训练集向量
    trainMat = []
    # 训练集分类
    trainClasses = []
    # 从训练集编号中按序读取（已删除被抽为样本的十个编号）
    for docIndex in trainingSet:
        # 调用文档向量转换函数setOfWords2Vec，利用词汇列表和对应训练集的文档内容获取文档向量
        # 并添加到训练集向量list中
        trainMat.append(setOfWords2Vec(vocablist, docList[docIndex]))
        # 添加这篇文章对应的分类到训练集分类中
        trainClasses.append(classList[docIndex])
    # 调用训练函数trainNB0，使用文档向量集和标签集获取训练结果
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 错误计数
    errorCount = 0
    # 遍历测试集
    for docIndex in testSet:
        # 获取测试集的文档向量
        wordVector = setOfWords2Vec(vocablist, docList[docIndex])
        # 进行贝叶斯过滤，并和真值进行校验
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('错误率为：',errorCount/len(testSet))
    return errorCount/len(testSet)


# 高频词统计器
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    # 统计词汇表中每个词在总内容表里出现的次数
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    # 统计完后进行排序
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse = True)
    # 不移除高频关键词时，错误率大约在0.2~0.5
    # return []
    # 移除前30个高频词后，错误率也在0.2~0.5之间，但移除后结果更稳定，不移除时出现0.1或0.6等极端情况的概率更大
    return sortedFreq[:10]

def getStopwordList():
    StopwordList = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't",
            "as","at","be","because","been","before","being","below","between","both","but","by","can't",
            "cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down",
            "during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't",
            "having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself",
            "his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's",
            "its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off",
            "on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same",
            "shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that",
            "that's","the","their","theirs","them","themselves","then","there","there's","these","they",
            "they'd","they'll","they're","they've","this","those","through","to","too","under","until",
            "up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what",
            "what's","when","when's","where","where's","which","while","who","who's","whom","why","why's",
            "with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours",
            "yourself","yourselves"]
    return StopwordList

#  区域倾向分类测试
def localWords(feed1, feed0):
    # 文档列表、标签列表、全词汇合集
    docList = []
    classList = []
    fullText = []
    # 测试RSS源的正文从'entries'标签下开始
    # 获取两个RSS源里最短的长度
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    # 获取RSS正文
    for i in range(minLen):
        # entries标签下为正文，i确定条目，summary标签下是每条的正文
        wordList = textParse(feed1['entries'][i]['summary'])
        # 将获取到的每条正添加到文档列表、全列表中
        docList.append(wordList)
        fullText.extend(wordList)
        # 同时置标签列表为1
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 以下增加删除stopwords部分
    stopwordList = getStopwordList()
    for doc in docList:
        for word in doc:
            if word in stopwordList:
                doc.remove(word)
    for word in fullText:
        if word in stopwordList:
            fullText.remove(word)
    # 以上增加删除stopwords部分

    # 调用createVocabList创建词汇表
    vocabList = createVocabList(docList)
    # 调用calcMostFreq获取出现频率最高的前XX个词汇
    top30Word = calcMostFreq(vocabList, fullText)
    # 从词汇表中删除高频词汇
    for pairW in top30Word:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    # 训练集，长度为2倍最小长度
    trainingSet = list(range(2 * minLen))
    # 测试集
    testSet = []
    # 随机抽取20个做测试集
    # 实际操作中有可能出现数据总共都没有20条的情况
    for i in range(20):
        randIndex = int (random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        trainingSet.pop(randIndex)
    # 训练向量、训练标签
    trainMat = []
    trainClasses = []
    # 调用bagOfWords2VecMN（词袋型向量转换函数）将测试集转换为向量并且输入到训练向量集中
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        # 并记录其所属的标签
        trainClasses.append(classList[docIndex])
    # 进行训练
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 错误计数
    errorCount = 0
    # 遍历测试集
    for docIndex in testSet:
        # 调用bagOfWords2VecMN（词袋型向量转换函数）将测试集转换为向量
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # 调用分类函数分类，并验证结果
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('错误率为：', errorCount/len(testSet))
    # return errorCount/len(testSet)
    return vocabList, p0V, p1V

# 对去除高频词和停用词的测试结果的绘图函数
def createPlot(aResult):
    import matplotlib.pyplot as plt
    # 生成一个图表对象
    fig = plt.figure()
    # 在图表上添加子图，三个参数分别为子图行数、子图列数、子图位置
    ax = fig.add_subplot(111)
    # 参数为x，y，大小，颜色，标记......
    # 在子图中绘制数据的后两项，datingDataMat[:, 1]表示逐行取之后取其第二项
    # 之后为15+标记作为各点的大小和颜色，用以区分点
    ax.bar(left = arange(100),height=aResult, width = 1,color="lightblue")
    # 绘图
    plt.show()

# 获取最具表征性的词汇
def getTopWord(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    # 书中使用的权重为-6，会输出大量的词
    for i in range(len(p0V)):
        if p0V[i] > -4.5:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -4.5:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key = lambda pair:pair[1], reverse = True)
    print('以下为旧金山市的高频词')
    for item in sortedSF:
        print(item[0])
    print('以下为纽约市的高频词')
    sortedNY = sorted(topNY, key = lambda pair:pair[1], reverse = True)
    for item in sortedNY:
        print(item[0])

if __name__ == '__main__':
    # listOPosts, listClasses = loadDataSet()
    # print(listOPosts)
    # myVocabList = createVocabList(listOPosts)
    # # 和书上输出顺序不同（顺序随机），但内容相同
    # print(myVocabList)
    # # 由于myVocabList的输出顺序随机，所以这里输出结果也随机
    # print(setOfWords2Vec(myVocabList,listOPosts[0]))
    # 构建文档向量list
    # trainMat = []
    # for postinDoc in listOPosts:
    #     # 逐个获取每篇文档的文档向量，放入trainMat中
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # p0v, p1v, pAb = trainNB0(trainMat, listClasses)
    # # print(pAb)
    # print(p0v)
    # testingNB()
    # spamTest()
    # count = 0
    # for i in range(10):
    #     count += spamTest()
    # print(count/10)
    ny = feedparser.parse('https://newyork.craigslist.org/search/tfr?format=rss')
    sf = feedparser.parse('https://sfbay.craigslist.org/search/tfr?format=rss')
    # vocabList, pSF, pNY = localWords(ny, sf)
    # 启用以下测试语句需要修改localWords的输出语句
    # result = []
    # for i in range(100):
    #     result.append(localWords(ny, sf))
    # result.sort()
    # aResult = array(result)
    # createPlot(aResult)
    getTopWord(ny,sf)



