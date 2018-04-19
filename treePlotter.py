# -*- coding: utf-8 -*-

import trees

import matplotlib.pyplot as plt

# 定义matplotlib绘图的文本框和箭头格式，还有他妈的定义字体
# 定义字体
plt.rcParams['font.sans-serif']=['SimHei']
# 用于正常显示负号
plt.rcParams['axes.unicode_minus']=False
# boxstyle：文本框风格，sawtoot：锯齿；round4：圆角4
# fc：背景灰度，0黑，1白
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
# 箭头格式
arrow_args = dict(arrowstyle='<-')

# 新建图形
def createPlot(inTree):
    # 创建图形实例，编号1，底色为white
    fig = plt.figure(1, facecolor= 'white')
    # 清空当前图形实例
    fig.clf()
    # 不显示坐标轴及其数值
    axprops = dict(xticks = [], yticks = [])
    # 指定子图规格，111为一行一列的子图划分方式的第一个位置，frameon：是否显示边框
    # createPlot.ax1为绑定之后创建的子图
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    # 储存叶节点数目（宽度）
    plotTree.totalW = getNumLeafs(inTree)
    # 储存深度（高度）
    plotTree.totalD = getTreeDepth(inTree)
    # 定义初始x,y
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1

    plotTree(inTree, (0.5, 1), '')
    plt.show()

def createPlot2():
    # 创建图形实例，编号1，底色为white
    fig = plt.figure(1, facecolor= 'white')
    # 清空当前图形实例
    fig.clf()
    # 指定子图规格，111为一行一列的子图划分方式的第一个位置，frameon：是否显示边框
    # createPlot.ax1为绑定之后创建的子图
    createPlot.ax1 = plt.subplot(111, frameon = False)
    # 创建两个文字标注点，名称，箭头指向位置，箭头起始位置，节点类型
    plotNode('决策节点', (0.5, 0.1),(0.1, 0.5), decisionNode)
    plotNode('-叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


# 创建节点。名称，箭头指向位置，箭头起始位置，节点类型
def plotNode(nodeText, centerPt, parentPt, nodeType):
    # 文本、箭头起始位置、起始位置的数字的意义，axes fraction为占数轴的百分比
    createPlot.ax1.annotate(nodeText, xy = parentPt,xycoords = 'axes fraction',
    # 文本坐标（箭头指向位置）、指向坐标数字的意义、va和ha大概是指定文字居中，懒得扣细节了
    xytext = centerPt, textcoords = 'axes fraction',va = 'center',ha = 'center',
    # 文本框类型
    bbox = nodeType,
    # 指定箭头格式
    arrowprops = arrow_args)

def getNumLeafs(myTree):
    # 叶节点计数
    numLeafs = 0
    # 获取第一个关键字
    firstStr = list(myTree.keys())[0]
    # 获取第一个子树
    secondDict = myTree[firstStr]
    # 遍历子树
    for key in secondDict.keys():
        # 只要子树不是叶节点就递归向下
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    # 深度计数
    maxDepth = 0
    # 获取第一个关键字
    firstStr = list(myTree.keys())[0]
    # 获取第一个子树
    secondDict = myTree[firstStr]
    # 遍历子树
    for key in secondDict.keys():
        # 只要子树不是叶节点就递归向下
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{'无需浮出水面':{0:'no',1:{'有蹼':{0:'no',1:'yes'}}}},
                   {'无需浮出水面':{0:'no',1:{'有蹼':{0:{'有头':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]

# 在节点之间填充数字
def plotMidText(cntrPt, parentPt, txtString):
    # 取x和y的中间值
    xMid = (parentPt[0] - cntrPt[0])/2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2 + cntrPt[1]
    # 添加数字
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    # 获取当前树叶节点数目和树深度
    numLeafs = getNumLeafs(myTree)
    # 不知道这里拿深度有什么吊用，因为在创建函数里已经拿了深度了
    # 这之后的深度计算都是基于上层传下来的深度值再算的，另，测试把这句注销掉也不影响运行
    depth = getTreeDepth(myTree)
    # 获取当然树的根节点名
    firstStr = list(myTree.keys())[0]
    # 计算节点坐标
    cntrPt = (plotTree.xOff + (1 + numLeafs)/2/plotTree.totalW, plotTree.yOff)
    # 调用plotMidText创建数字
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 创建节点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 获取子树
    secondDict = myTree[firstStr]
    # 更新节点Y坐标，当前y - 1/节点深度即为新的一层节点的Y坐标
    plotTree.yOff = plotTree.yOff - 1/plotTree.totalD
    # 遍历子树
    for key in secondDict.keys():
        # 若仍为子树，则递归（这个决策树必然是二叉树）
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntrPt, str(key))
        # 到了叶节点则
        else:
            # 计算新x坐标
            plotTree.xOff = plotTree.xOff + 1/plotTree.totalW
            # 创建节点（叶）
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # 标记数字
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    # 这层结束时更新基准y坐标
    plotTree.yOff = plotTree.yOff + 1/plotTree.totalD


# 分类器
def classify(inputTree, featLables, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex= featLables.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if isinstance(secondDict[key], dict):
                classLable = classify(secondDict[key], featLables, testVec)
            else:
                classLable = secondDict[key]
    return classLable

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

if __name__ == '__main__':
    # createPlot2()
    # print(retrieveTree(1))
    # myTree = retrieveTree(0)
    # print(myTree)
    # print(getNumLeafs(myTree))
    # print(getTreeDepth(myTree))
    # myTree['无需浮出水面'][3] = 'maybe'
    # createPlot(myTree)
    # myDat, labels = trees.createDataSet()
    # print(myTree)
    # print(classify(myTree, labels, [1,0]))
    # print(classify(myTree, labels, [1,1]))
    fr = open('./data/ch03/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print()
    lensesLables = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = trees.createTree(lenses, lensesLables)
    print(lensesTree)
    createPlot(lensesTree)


