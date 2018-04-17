# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# 定义matplotlib绘图的文本框和箭头格式，还有他妈的定义字体
# 定义字体
plt.rcParams['font.sans-serif']=['SimHei']
# 用于正常显示负号
plt.rcParams['axes.unicode_minus']=False
# boxstyle：文本框风格，sawtoot：锯齿；round4：圆角4
# fc：背景灰度，0黑，1白
decisinoNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
# 箭头格式
arrow_args = dict(arrowstyle='<-')

# 新建图形
def createPlot():
    # 创建图形实例，编号1，底色为white
    fig = plt.figure(1, facecolor= 'white')
    # 清空一下当前图形实例
    fig.clf()
    # 指定子图规格，111为一行一列的子图划分方式的第一个微珠，frameon：是否显示边框
    # createPlot.ax1为绑定之后创建的子图
    createPlot.ax1 = plt.subplot(111, frameon = False)
    # 创建两个文字标注点，名称，箭头指向位置，箭头起始位置，节点类型
    plotNode('决策节点', (0.5, 0.1),(0.1, 0.5), decisinoNode)
    plotNode('-叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

# 创建节点，名称，箭头指向位置，箭头起始位置，节点类型
def plotNode(nodeText, centerPt, parentPt, nodeType):
    # 文本、箭头起始位置、起始位置的数字的意义，axes fraction为占数轴的百分比
    createPlot.ax1.annotate(nodeText, xy = parentPt,xycoords = 'axes fraction',
    # 文本坐标（箭头指向位置）、指向坐标数字的意义、va和ha大概是指定文字居中，懒得扣细节了
    xytext = centerPt, textcoords = 'axes points',va = 'center',ha = 'center',
    # 文本框类型
    bbox = nodeType,
    # 指定箭头格式
    arrowprops = arrow_args)

def getNumleafs(myTree):
    # 叶节点计数
    numLeafs = 0
    # 获取第一个关键字
    firstStr = list(myTree.keys())[0]
    # 获取第一个子树
    secondDict = myTree[firstStr]
    # 遍历子树
    for key in secondDict.keys():
        # 只要子树不是叶节点就递归向下
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumleafs(secondDict[key])
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
        if type(secondDict[key]).__name__ == 'dict':
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

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    pass

if __name__ == '__main__':
    # createPlot()
    print(retrieveTree(1))
    myTree = retrieveTree(0)
    print(getNumleafs(myTree))
    print(getTreeDepth(myTree))