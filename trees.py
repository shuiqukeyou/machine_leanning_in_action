import operator

from math import log


# 计算信息熵，信息熵越大，信息无序程度越高
def calcShannonEnt(dataSet):
    # 获取数据总数
    numEntrise = len(dataSet)
    # 创建记录所有标签及其出现次数的dict
    labelCounts = {}
    # 遍历数据，数据格式[1, 1, 'yes']：是否无需浮出水面，是否有蹼，是否为鱼类
    for featVec in dataSet:
        # 读取每项数据的最后一条
        currentLabel = featVec[-1]
        # 如果这个标签不存在于dict中则新建，然后置其值为0
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 每找到一个标签，标签计数+1
        labelCounts[currentLabel] += 1
    # 设信息熵初始值为0
    shannonEnt = 0
    # 遍历标签dict
    for key in labelCounts:
        # 计算每个标签出现的概率
        prob = labelCounts[key]/numEntrise
        # 按公式计算出每个标签的信息熵，使用累减时因为信息熵都是负数
        shannonEnt -= prob * log(prob, 2)
    # 返回信息熵
    return shannonEnt

# 按照指定特征划分数据集
# 待划分数据集、划分数据集的特征、需要返回特征的值
def splitDataSet(dataSet, axis, value):
    # 创建空数据集
    retDataSet = []
    # 按行读取数据集
    for featVac in dataSet:
        # 如果这项数据的指定维度的数据值==需要返回的值
        if featVac[axis] == value:
            # 将这行数据的（除指定维度之外的）各项值储存到reducedFeatVec中
            reducedFeatVec = featVac[:axis]
            reducedFeatVec.extend(featVac[axis + 1:])
            # 将抽取出的数据储存到待划分出的数据集中
            retDataSet.append(reducedFeatVec)
    return  retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # 取数据集的第一项用以确定数据项的长度，再减一即是特征项数量
    numFeatures = len(dataSet[0]) - 1
    # 计算这个数据集的数据熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最佳信息熵差值
    bestInfoGain = 0
    # 最佳划分特征，初值设为-1
    bestFeature = -1
    # 遍历每类特征
    for i in range(numFeatures):
        # 遍历数据集的每项，并将每项当前应遍历的特征值取出，组成一个list
        featList = [example[i] for example in dataSet]
        # 将这个list去重，每个特征值只保留一项
        uniqueVals =set(featList)
        # 临时信息熵，用于记录以不同类特征值计算出的不同信息熵
        newEntropy = 0
        # 遍历每项特征值
        for value in uniqueVals:
            # 尝试以该项特征值及其特征类别进行划分
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算划分出的组占数据集的百分比
            prob = len(subDataSet)/len(dataSet)
            # 按信息熵公式计算出当前划分模式下的信息熵，并求和
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 用数据集总信息熵减去当前划分模式下的信息熵，差值越大，
        # 则表明当前划分模式的信息熵越大，无序度越高，效果越差
        # 之所以用总信息熵倒一遍时因为分组信息熵理论上没有上限，但总是不会超过总信息熵
        infoGain = baseEntropy - newEntropy
        # 若使用当前划分方法获得的信息熵更小（infoGain值更大），则无序度更低，则更有效
        # 则记录当前划分方法的信息熵差值和划分方法
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 投票分类
def majorityCnt(classList):
    # 标签字典
    classCount = {}
    # 遍历标签
    for vote in classList:
        # 不存在则写入，并计数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    # 返回出现最多的分类的名称
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    # 读取各组分类
    classList = [example[-1] for example in dataSet]
    # 如果当前分类的第一项的计数==当前分类长度，则已完成分类，返回标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 若已进行到最后一项仍未完成分类
    if len(dataSet[0]) == 1:
        # 进行分类投票
        return majorityCnt(classList)
    # 选择最好的数据集划分方式
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 取出当前划分方式的对应的标签
    bestFeatLabel = labels[bestFeat]
    # 使用标签开始建树
    myTree = {bestFeatLabel:{}}
    # 在标签集中删除已被划分的标签
    # 划分函数会删除划分轴，故标签组也需要删除掉已经划分好了的标签
    labels.pop(bestFeat)
    # 按照最好的数据集划分方式提取出数据值
    featValues = [example[bestFeat] for example in dataSet]
    # 数据值去重
    uniqueVals = set(featValues)
    # 遍历数据值
    for value in uniqueVals:
        # 复制标签组（已被删除掉已确定的部分）
        subLabels = labels[:]
        # 使用splitDataSet，根据划分方式和value进行划分，传入的标签组是删除已被划分标签的标签组
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['无需浮出水面', '有蹼']
    return dataSet, labels

if __name__ == '__main__':
    myDat, labels = createDataSet()
    print(myDat)
    # print(calcShannonEnt(myDat))
    # print(splitDataSet(myDat, 0, 1))
    # print(chooseBestFeatureToSplit(myDat))
    print(createTree(myDat,labels))
