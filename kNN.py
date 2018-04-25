import operator

from os import listdir
from numpy import *


def createDataSet():
    group = array([[1.0,1.1],
                   [1.0,1.0],
                   [0,0],
                   [0,0.1]]) # numpy.array(定义numpy数组)，此处为一个二维矩阵（四行两列）
    labels = ['A','A','B','B']
    return group, labels



# （输入向量，训练集，标签向量，选择最近值的数目）
def classfy0(inX, dataSet, labels, k):
    # numpy.shape(返回矩阵的行数和列数)
    dataSetSize = dataSet.shape[0]
    # numpy.tile(生成矩阵),用inX（输入向量）和列数生成一个各项均为输入向量，且行数和样本集相同的数组（这里为行列均相同）
    # 再用得到矩阵减去样本集，得到输入向量和每组向量值的差值矩阵
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # 对差值矩阵平方，在numpy下即为对矩阵各项平方
    sqDiffMat = diffMat ** 2
    # sum（对各行求和），numpy矩阵才有的方法，axis为求和的维数，为0时对列求和（返回2项），为1时对行求和（返回4项），依次类推
    # 此处为axis = 1，即每行求和，即返回各差值的平方和数组
    sqDistance = sqDiffMat.sum(axis = 1)
    # 对各差值的平方和数组各项开方，最终结果为输入向量和样本集各项的距离（平方和开根号）数组
    distances = sqDistance ** 0.5
    # 以上为距离计算（个值差平方和开根号）

    # 对输入向量和样本集各项的距离数组从小到大进行排序,返回结果为从小到大项在原数组中的位置
    # [2,4,0,1] 会返回 [2,3,0,1]，即最小的是第2个，第二小的是第3个...以此类推
    sortedDistIndicies = distances.argsort()
    # 新建空dict
    classCount = {}
    # 取前k项
    for i in range(k):
        # 按距离顺序数组从标签向量里取出这个距离差对应的标签，比如最小的是第二个，则去类别标签中取第二个的标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在dict中按标签获取值，没有时该标签时获取到0，+1后写回dict
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # sorted()排序，需要传入一个可迭代对象（classCount.items()），比较方法（cmp，此处使用默认方法），
    # 关键字（key，此处使用operator.itemgetter(1)，返回每项的第二个值作为key），
    # reverse True为降序，False为升序
    # 最终返回一个list，其中每项为一个tuple
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    # 最终返回出现频率最高的类别的标签
    return sortedClassCount[0][0]



# 装载文件，返回数据矩阵和标签列表
def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 按行读取整个为一个list，另，使用file.readline()时为返回迭代器，一次读取一行
    arrayOLines = fr.readlines()
    # 返回行数
    numberOfLines = len(arrayOLines)
    # 返回行数与numberOfLines相同。列数为3的0矩阵（储存数据用）
    returnMat = zeros((numberOfLines, 3))
    # 标签数组
    classLabeLVector = []
    index = 0
    for line in arrayOLines:
        # 删除每行中的空白内容
        line = line.strip()
        # 按空白将每行分割开，返回一个list
        listFromLine = line.split('\t')
        # 将每行数据的前三项按行写入空矩阵中
        returnMat[index, :] = listFromLine[0:3]
        # 将标签写入到标签数组中
        classLabeLVector.append(int(listFromLine[-1]))
        index += 1
    # 返回数据矩阵和每行数据对应的标签
    return returnMat, classLabeLVector



# 归一化
def autoNorm(dataSet):
    # 以第一列为基准，选出矩阵中最小的行
    minVals = dataSet.min(0)
    # 以第一列为基准，选出矩阵中最大的行
    maxVals = dataSet.max(0)
    # 将矩阵中的最大的行和最小的行作差
    ranges = maxVals - minVals
    # 按照dataSet的行数和列数构建0矩阵
    noreDataSet = zeros(shape(dataSet))
    # 获取数据集行数m
    m = dataSet.shape[0]
    # 先使用tile生成一个m行，每行为最小行（minVals）的矩阵
    # 然后使用原数据集减去这个矩阵，得到原数据集和最小项的差值矩阵
    normDataSet = dataSet - tile(minVals, (m,1))
    # 先使用tile生成一个m行，每行为最大行（maxVals）的矩阵
    # 然后令差值矩阵的每项除以最大行矩阵的对应项，完成归一化
    normDataSet = normDataSet/tile(ranges, (m,1))
    # 返回归一化后的矩阵、差值、最小行
    return normDataSet, ranges, minVals



def datingClassTest():
    # 取数据的百分比
    hoRatio = 0.1
    # 装载数据
    datingDataMat, datingLabals = file2matrix('./data/ch02/datingTestSet2.txt')
    # 归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获取总行数
    m = normMat.shape[0]
    # 总行数*百分比 = 取的训练集的实际数量
    numTestVecs = int(m * hoRatio)
    # 错误计数
    errorCount = 0
    for i in range(numTestVecs):
        # normMat[i,:]（取矩阵的第i行做输入向量，一直取到numTestVecs-1）
        # normMat[numTestVecs:m,:]（从numTestVecs取到矩阵末尾，做训练集）
        # datingLabals[numTestVecs:m]（取从numTestVecs取到矩阵末尾的标签序列）
        classifierResult = classfy0(normMat[i,:], normMat[numTestVecs:m,:], datingLabals[numTestVecs:m], 3)
        print('分类器返回结果为：%d,真实结果为：%d'%(classifierResult, datingLabals[i]))
        # 对分类器结果和真实标签进行对比
        if (classifierResult != datingLabals[i]):
            errorCount += 1
    print("总错误率为：%f"%(errorCount/numTestVecs))



def classifyPerson():
    # 标签列表
    resultList = ['毫无兴趣', '略有兴趣', '兴趣很大']
    # 以下数据录入
    percentTats = float(input('花费在游戏上的时间百分比？'))
    ffMile = float(input('每年的飞行里程？'))
    iceCream = float(input('每年消费的冰淇淋量？'))
    # 以上数据录入
    # 装载样本集
    datingDataMat, datingLabels = file2matrix('./data/ch02/datingTestSet2.txt')
    # 归一化样本集
    #返回归一后样本集、差值、最小行
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 将录入的三项数据创建为一个单行矩阵
    inArr = array([ffMile, percentTats, iceCream])
    # 调用分类函数classfy0，将数据项归一化（(inArr-minVals)/ranges）后传入
    classiferResult = classfy0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('你很可能是这种人：', resultList[classiferResult - 1])

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


def handwritingClassTest():
    # 空标签集
    hwLabels = []
    # 读取训练集文件列表
    trainingFileList = listdir('./data/ch02/trainingDigits')
    # 获取训练集文件数目
    m = len(trainingFileList)
    # 按照训练集文件数目，创建一个m行1024列的0矩阵
    trainingMat = zeros((m, 1024))
    # 加载训练集
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 除去训练集文件名的扩展名
        fileStr = fileNameStr.split('.')[0]
        # 获取训练集文件名中的真值
        classNumStr = int(fileStr.split('_')[0])
        # 将训练集文件的真值写入标签中
        hwLabels.append(classNumStr)
        # 调用img2vecetor将训练集装载为向量，并写入训练集矩阵中
        trainingMat[i, : ] = img2vecetor('./data/ch02/trainingDigits/%s'% fileNameStr)
    # 打开测试集文件，以下几乎同上
    testFileList = listdir('./data/ch02/testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        # 转换测试集文件为向量
        vectorUnderTest = img2vecetor('./data/ch02/testDigits/%s'% fileNameStr)
        # 调用classfy0进行分类
        classifierResult = classfy0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('这个样本被预测为：%d，实际上是%d'%(classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1
    print('\n错误总数为%d' % errorCount)
    print('\n总错误率为%f' % (errorCount/mTest))


if __name__ == '__main__':
    # group, lables = createDataSet()
    # temp = classfy0([0,0], group, lables, 3)
    # print(temp)
    datingDataMat, datingLabals = file2matrix('./data/ch02/datingTestSet2.txt')
    # print(datingDataMat)
    # print(datingLabals[0:20])
    import matplotlib.pyplot as plt
    # 生成一个图表对象
    fig = plt.figure()
    # 在图表上添加子图，三个参数分别为子图行数、子图列数、子图位置
    ax = fig.add_subplot(111)
    # 参数为x，y，大小，颜色，标记......
    # 在子图中绘制数据的后两项，datingDataMat[:, 1]表示逐行取之后取其第二项
    # 之后为15+标记作为各点的大小和颜色，用以区分点
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabals), 15.0 * array(datingLabals))
    # 绘图
    plt.show()
    # normMat, ranges, minVals = autoNorm(datingDataMat)
    # print(normMat)
    # datingClassTest()
    # classifyPerson()
    # textVector = img2vecetor('./data/ch02/testDigits/0_13.txt')
    # print(textVector[0, 0:31])
    # handwritingClassTest()