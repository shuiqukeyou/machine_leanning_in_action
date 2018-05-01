from numpy import *


# 创建数据集
def loadSimpData():
    dataMat = matrix([[1, 2.1],
                      [2, 1.1],
                      [1.3, 1],
                      [1, 1],
                      [2, 1]])
    classLabels = [1, 1, -1, -1, 1]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1
    return retArray


# 数据集、标签、权重向量
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10 # 步长
    bestStump = {} # 初始空单层决策树
    bestClasEst = mat(zeros((m, 1))) # 初始权重全为1
    minError = inf # 初始化最小错误率为无穷大
    for i in range(n):# 遍历数据集的所有特征，通过最大值和最小值决定步长
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, numSteps+1): # 遍历所有步长
            for inequal in ['lt', 'gt']: # 在大于和小于之间切换
                threshVal = rangeMin + float(j) * stepSize
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) # stumpClassify的结果会根据inequal的变化而变化
                errArr = mat(ones((m, 1))) # 将初始错误率全设为1
                errArr[predictedVals == labelMat] = 0 # 如果有预测正确的，将其错误率设为0
                weightedError = D.T * errArr # 用权重矩阵 * 错误率矩阵
                # print('split:dim %d, thresh %.2f, thresh inqual:%s,误差加权：%.3f' % (i, threshVal, inequal, weightedError))
                if weightedError < minError: # 如果新错误率小于之前的错误率，则将错误矩阵、权重矩阵、决策树进行更新
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst # 返回最佳单层决策树，最小错误率，类别估计值


# 完整Adaboost
# 数据集、类别标签、迭代次数（分类器数目）
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = [] # 弱分类器列表
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m) # 权重矩阵
    aggClassEst = mat(zeros((m, 1))) # 分类结果
    for i in range(numIt): # 开始进行迭代
        bestStump, error, classEst = buildStump(dataArr, classLabels, D) # 进行一次弱分类
        # print('D:',D.T)
        alpha = float(0.5 * log((1 - error) / max(error, 1e-16))) # 计算本次弱决策树输出结果的权重，error越高权重越低
        bestStump['alpha'] = alpha # 向最佳单层决策树中添加计算出的alpha值
        weakClassArr.append(bestStump)# 并将这棵最佳单层决策树添加到弱分类器列表中
        # print('classEst:',classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)# -1 * 实际标签 * 预测标签（只有预测错才得1）
        D = multiply(D, exp(expon)) # 权重矩阵更新
        D = D/D.sum() # 权重矩阵更新
        aggClassEst += alpha * classEst
        # print('aggClassEst:', aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1))) # 统计总错误
        errorRate = aggErrors.sum()/m # 统计错误率
        # print('总错误率：', errorRate)
        if errorRate == 0: # 若错误率为0则退出循环
            break
    return weakClassArr, aggClassEst



def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print (aggClassEst)
    return sign(aggClassEst)


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat -1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1, 1)
    ySum = 0
    numPosClas = sum(array(classLabels) == 1)
    ySetp = 1/numPosClas
    xStep = 1/(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1:
            delX = 0
            delY = ySetp
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('假阳率')
    plt.ylabel('真阳率')
    plt.title('马疝气的adaboost检测系统的ROC曲线')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('曲线下的面积是：',ySum * xStep)



if __name__ == '__main__':
    # dataMat, classLabels = loadSimpData()
    # D = mat(ones((5, 1)) / 5)
    # bestStump, minError, bestClasEst = buildStump(dataMat, classLabels, D)
    # classifierArr = adaBoostTrainDS(dataMat, classLabels, 30)
    # print(adaClassify([[5, 5], [0, 0]], classifierArr))
    dataArr, labelArr = loadDataSet('./data/ch07/horseColicTraining2.txt')
    classifierArray, aggClassEst= adaBoostTrainDS(dataArr, labelArr, 50)
    testArr, testLabelArr = loadDataSet('./data/ch07/horseColicTest2.txt')
    prediction10 = adaClassify(testArr, classifierArray)
    errArr = mat(ones((67,1)))
    print(errArr[prediction10 != mat(testLabelArr).T].sum()/67)
    # plotROC(aggClassEst.T, labelArr)
