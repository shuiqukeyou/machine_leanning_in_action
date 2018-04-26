
import matplotlib.pyplot as plt

from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('./data/ch05/testSet.txt')
    for line in fr.readlines():
        # 去除每行的空内容，并对其进行分割
        lineArr = line.strip().split()
        # 这里加强制转换是因为直接读出来的东西都是字符串
        dataMat.append([1, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1/(1 + exp(-inX))

# 梯度上升(每次上升和所有变量做运算)
def gradAscent(dataMatIn, classLables):
    # 将dataMatIn转换为numpy矩阵
    dataMatrix = mat(dataMatIn)
    # 将classLables转换为numpy矩阵，并将其转置
    labelMat = mat(classLables).transpose()
    # 返回行数列数
    m, n = shape(dataMatrix)
    # 回归系数（决定每次回归的增量大小）
    alpha = 0.001
    # 迭代次数上限
    maxCycles = 500
    # 创建一个单位列矩阵
    weights = ones((n, 1))
    for k in range(maxCycles):
        # 用sigmoid函数计算当前weights下的阶跃程度
        h = sigmoid(dataMatrix * weights)
        # 计算正式值与当前阶跃度的差
        error = (labelMat - h)
        # 利用差值和回归系数迭代修正weights
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 绘图
def plotBestFit(weights):
    # 获取数据集和标签
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    # 获取项数
    n = shape(dataArr)[0]
    # 数据集坐标集合
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    # 遍历数据
    for i in range(n):
        # 若数据属于内部1
        if int(labelMat[i])  == 1:
            # x、y坐标集合增加
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    # 绘图部分
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker='s')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    # 指定XY坐标轴范围
    x = arange(-3, 3, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    # 绘制标签
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 随机梯度上升
# 每次和一个点进行运算（这个版本还不是随机点），精度差，收敛后期抖动大
# 固定梯度，收敛速度慢
def stocGradAscent0(dataMatrix, classLables):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLables[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 随机梯度上升
# 每次和一个随机点进行运算
# 梯度渐减，加快收敛速度（梯度永不为0，保证多次迭代后新数据对模型仍然有影响）
def stocGradAscent1(dataMatrix, classLables, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 梯度渐减
            alpha = 4/(1+ j + i ) + 0.01
            # 随机选取参照点
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLables[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 移除参照点，避免将其重复作为参照点
            dataIndex.pop(randIndex)
    return weights

# 分类函数
def classifVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def colicTest():
    frTrain = open('./data/ch05/horseColicTraining.txt')
    frTest = open('./data/ch05/horseColicTest.txt')
    trainingSet = []
    trainingLables = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLables.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLables, 500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = errorCount/numTestVec
    print('错误率为:',errorRate)
    return errorRate

# 入口函数
def multiTest():
    # 共测试十次，求平均错误率
    numTests = 10
    errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
    print('在经过%d轮迭代后平均错误率为：%f'% (numTests, errorSum/numTests))

if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # weights = gradAscent(dataArr, labelMat)
    # plotBestFit(weights.getA())
    # weights = stocGradAscent0(array(dataArr), labelMat)
    # weights = stocGradAscent1(array(dataArr), labelMat)
    # plotBestFit(weights)
    multiTest()
