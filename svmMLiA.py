from numpy import *

# 读取数据
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    # 返回数据矩阵和标签矩阵
    return dataMat, labelMat


#
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj,H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 简化版SOM
# 数据集、标签集、常数C、容错率、最大迭代次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 数据集和标签集转置
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    # 公式中用到的一个重要常数项
    b = 0
    # 行列数
    m,n = shape(dataMatrix)
    # 创建初始alphas，初始值为m行的单列矩阵
    alphas = mat(zeros((m, 1)))
    # 迭代次数
    iter = 0
    # 只要迭代次数少于设定的迭代上限，就继续进行遍历
    while (iter < maxIter):
        # 每轮迭代前将标记值记0，0表示alpha尚未进行优化
        alphaPairsChanged = 0
        for i in range(m):
            # 预测类别
            # multiply为矩阵各项相乘（不是矩阵乘法），获得由alphas处理过后的标签矩阵，在对其进行转置
            # dataMatrix * dataMatrix[i, : ].T：用数据矩阵 X 当前行的转置
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, : ].T)) + b
            # 计算预测的类别和实际值的差值
            Ei = fXi - float(labelMat[i])
            # 如果对应标签 * 预测值得绝对值大于容错率（分别对应正容差和负容差）
            # 并且对应情况的alphas[i]没有超过传入的常数C，则可以进行优化
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 由selectJrand随机选择i~m间的一个值将其和i组成值对
                j = selectJrand(i ,m)
                # 同样计算j的预测类别
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                # 计算j的预测值的误差
                Ej = fXj - float(labelMat[j])
                # 为了避免多重引用，将alphas中对应的i和j值备份
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 计算L和H，i和j对应的标签相同和不相同时需要分别处理
                # L和H为这个数据对和分隔超平面的距离
                if (labelMat[i] != labelMat[j]):
                    # 若不为同类，则这两个点应该各自在分隔超平面的两侧
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    # 若为同类，则这两个点应该各自在分隔超平面的同侧
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果计算出的L和H相等，则表明分隔超平面正好在中间，终止本次循环
                if L == H:
                    print('L == H')
                    continue
                # 计算alpha[j]的最优修改量
                eta = 2 * dataMatrix[i, : ] * dataMatrix[j, : ].T - dataMatrix[i, : ] * dataMatrix[i, : ].T - dataMatrix[j, : ] * dataMatrix[j, : ].T
                # 简化部分
                # 实际上当eta值为0时，需要退出这轮for循环
                if eta >= 0:
                    print('eta>=0')
                    continue
                # 计算新alphas[j]值
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 当alphas[j]和旧alphas[j]值差值极小时，直接直接终止本次循环，不对alphas[i]值进行调整
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('alphas[j]变换不够充分')
                    continue
                # 对alphas[i]进行同值但反向的调整
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 更新常数项
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, : ] * dataMatrix[i, : ].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, : ] * dataMatrix[j, : ].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, : ] * dataMatrix[j, : ].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, : ] * dataMatrix[j, : ].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2
                # 成功的改变了一对alpha，变化标记+1
                alphaPairsChanged += 1
                print('第%d次迭代  在第%d项时，当前数据对改变%d次'%(iter, i, alphaPairsChanged))
        # 退出上方循环后判断是否完成了一轮迭代
        # 如果标记值没有变动，则完成了一次收敛，迭代次数+1
        if (alphaPairsChanged == 0 ):
            iter += 1
        else:
            iter = 0
        print('迭代次数: %d' % iter)
    return b, alphas


# 实质上是利用这个类做一个数据结构
# 对应简化版SOM函数中的各项重要参数的初始化
class optStruct(object):
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


# 计算给定值的E值
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 计算J的最佳改变值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    # 返回非0E值对应的alpha值
    validEcacheList = nonzero(oS.eCache[ : ,0 ].A)[0]
    # 若有多个可用的alpha值，则进行遍历
    if len(validEcacheList) > 1:
        # 遍历，选出使得改变最大的值
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 随机选择一个
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 计算误差并存入缓存
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

# 完整SOM函数中的内循环部分
def innerL(i, oS):
    Ei= calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L == H')
            return 0
        eta = 2 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print('eta > 0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('alphas[j]值变换不够充分')
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0


# 完整SOM函数的外循环部分
def smoP(dataMatIn, classLables, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLables).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print('fullSet，第%d次迭代  在第%d项时，当前数据对改变%d次'%(iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound,第%d次迭代  在第%d项时，当前数据对改变%d次'%(iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
            print('迭代次数：%d'% iter)
    return oS.b, oS.alphas


# 核转换函数
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('我他娘遇到了一个错误：这个内核无法识别')
    return K



def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('./data/ch06/testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    # 进行训练
    print('there are %d Support Vectors' % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('训练错误率为%f'% (errorCount/m))
    # 进行测试
    dataArr, labelArr = loadDataSet('./data/ch06/testSetRBF2.txt')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, : ], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('测试错误率为：%f' % (errorCount/m))


# 装载数据
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, : ] = img2vecetor('%s/%s' % (dirName, fileNameStr))
    print('装载结束')
    return trainingMat, hwLabels

def testDigits(kTup = ('rbf',10)):
    dataArr, laberArr = loadImages('./data/ch06/trainingDigits')
    print('训练开始')
    b, alphas = smoP(dataArr, laberArr, 200, 0.0001, 10000, kTup)
    dataMat = mat(dataArr)
    labelMat = mat(laberArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print('共有%d个支持向量' % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(laberArr[i]):
            errorCount += 1
    print('本次训练错误率为%f' % (errorCount/m))
    dataArr, laberArr = loadImages('./data/ch06/testDigits')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(laberArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(laberArr[i]):
            errorCount += 1
    print('本次测试错误率为%f' % (errorCount / m))


# 装载图片二值化文件为向量
def img2vecetor(filename):
    # 创建1 X 1024的numpy数组
    returnVect = zeros((1, 1024))
    # 打开数据文件
    fr = open(filename)
    # 循环读取前32行（数据为32 X 32）
    for i in range(32):
        # 读取第i行
        lineStr = fr.readline()
        # 循环读取第i行的每个字符，并将其储存到numpy数组中
        for j in range(32):
            returnVect[0, 23*i+j] = int(lineStr[j])
    # 全部读取完成后返回结果
    return returnVect


if __name__ == '__main__':
    # dataArr, labelArr = loadDataSet('./data/ch06/testSet.txt')
    # print(labelArr)
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # print(b)
    # print(alphas[alphas>0])
    # b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    # print(b)
    # testRbf()
    testDigits(('rbf', 10))
