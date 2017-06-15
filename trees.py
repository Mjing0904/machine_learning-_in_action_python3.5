# 决策树的划分节点如果是最佳的特征组合呢？能否为每一个类别建立与之匹配的最佳特征组合？
# 最佳特征组合的特征权重又该怎么分配？信息增益率。利用信息熵不用考虑特征权重？

# 3.1 计算给定数据集的类别信息熵
from math import log
import operator
import numpy as np


def calcShannonEnt(dataSet):   # 计算决策属性熵，决策属性可以有多个值
    # 分割数据集后dataSet用retDataSet代替，递归计算某一特征值分类后的信息熵
    numEntries = len(dataSet)   # 分母
    labelCounts = {}   # 属性：信息增益
    for featVec in dataSet:
        currentLabel = featVec[-1]   # featVec[列号]，这样就可以计算每一列的信息熵了。
        # voteIlabel在classCount里，输出classCount[voteIlabel]（数量，否则为0）
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
        #  字典有自加功能？
        # if currentLabel not in labelCounts.keys():
        #     labelCounts[currentLabel] = 0
        # labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def calcShannonEnt1(dataSet):   # 计算决策属性熵，决策属性可以有多个值
    # 分割数据集后dataSet用retDataSet代替，递归计算某一特征值分类后的信息熵
    # 函数功能：计算：sum(xi*log2xi),xi为当前数据子集的类别对应的样本数。
    # numEntries = len(dataSet)   # 分母
    labelCounts = {}   # 属性：信息增益
    for featVec in dataSet:
        currentLabel = featVec[-1]   # featVec[列号]，这样就可以计算每一列的信息熵了。
        # voteIlabel在classCount里，输出classCount[voteIlabel]（数量，否则为0）
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
        #  字典有自加功能？
        # if currentLabel not in labelCounts.keys():
        #     labelCounts[currentLabel] = 0
        # labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # prob = float(labelCounts[key])/numEntries
        shannonEnt += labelCounts[key] * log(labelCounts[key], 2)
    return shannonEnt

# 创建一个简单的测试数据集，实际应用要用csv格式文件！
# 处理csv格式文件
# my_matrix = numpy.loadtxt(open("c:\\1.csv","rb"),delimiter=",",skiprows=0)
# numpy.savetxt('new.csv', my_matrix, delimiter = ',')
# txt转为csv，注意：1.数据间英文逗号间隔 2.每行回车换行！ 之后，改后缀名为.csv


def file3matrix(filename):
    # dataSet是no.array()格式？和列表同样的下标使用方式。如取第i个样本：dataSet[0]
    # 还需要特征的名称...这个怎么处理？？？
    dataSet = np.loadtxt(open(filename, "r"), delimiter=",", skiprows=0)
    # classLabelVector = returnMat[:, -1]
    # returnMat = returnMat[:, 0:-1]
    return dataSet


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 3.1.2 划分数据集
# 3-2 按照给定特征划分数据集


def splitDataSet(dataSet, axis, value):
    # 待划分的数据集、划分数据集的特征(特征序号，与dataset中一致)，对应特征的值
    # 参数对应：dataset、0（outlook）、'sunny'
    # 某个特征取值集合你事先知道？你要计算的是某一特征所有取值的信息熵之和，你怎么解决？循环调用。
    retDataSet = []   # python中传递的是列表的引用，一改全改，所以建立一个空列表做备用，为了不改变原始数据集。
    # 搜索到特征
    for featVec in dataSet:
        # 如果以所有特征集为键建立字典，是不是就可以用字典的内置函数来找Key，就不用for循环了？
        # 如果我把函数整合到一个函数中，就不用搜索了吧？
        if featVec[axis] == value:
            # 此处列表为什么使用切片删除?因为，每一个样本是列表形式，切片切除不会改变原来的样本，即featVec！
            # 在同一级上，对Outlook的字段sunny划分数据集后，还要按照outlook的其他字段如rain、overcast等字段划分数据集！
            # 这时候，就体现出数据集不能被改变的要求来！
            # 去掉'sunny'特征后的数据集
            reducedFeatVec = featVec[:axis]   # featVec[:, axis]
            # reducedFeatVec = featVec[axis+1, :]
            reducedFeatVec.extend(featVec[axis+1:])
            # 用append是因为列表可以做为列表的元素，即列表的嵌套。好处：一个列表元素是一个样本
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 接下来，循环计算信息熵和splitDataSet()函数，找到最好的特征划分方式
# 但是，要计算具有较多值的特征信息熵，所以也要信息熵归一化。思考怎么改代码！
# 3-3 选择最好的数据集划分方式
# 数据集要求：1.数据必须是由列表元素构成的列表，且所有列表元素长度相同。2、列表最后一列必须是类别标号。
# 实际中，要先填充缺失值。查询：缺失值的处理方法。
# 类的设计 函数参数的设计 你要思悟，怎样写出高效、风格好的代码来！


def chooseBestFeatureToSplit(dataSet):   # 选择出最好的特征，1.原始计算方法改成C4.5
    numFeatures = len(dataSet[0]) - 1
    # 计算决策属性熵
    baseEntropy = calcShannonEnt(dataSet)   # 类别熵
    # 计算最佳信息增益
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历每一个特征
    for i in range(numFeatures):
        # 首先获取某一特征的所有可能取值集合
        # 从列表中创建集合是python获得列表中唯一元素值得最快方法
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        # 属性字段信息熵
        newEntropy = 0.0
        # PunishnewEntropy = 0.0
        featEntropy = 0.0
        for value in uniqueVals:
            # 获得outlook的sunny的分类子集，子集分类结果>=2
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算sunny的分割信息熵，注意sunny分类子集所占比例的不同，如5/14,4/14,5/14
            probfeat = len(subDataSet)/len(dataSet)   # sunny的分类子集的占比
            # newEntropy += probfeat*calcShannonEnt(subDataSet)   # 这个需要乘多次...改！
            # 此处使用原来的方法分离决策树，改成C4.5，就要除以当前特征的分离信息
            featEntropy -= probfeat * log(probfeat, 2)
            newEntropy += probfeat * calcShannonEnt(subDataSet)
            # newEntropy *= probfeat
            # print("分母是： %f" %newEntropy)
            # 要惩罚具有较多特征取值的特征！信息熵归一化，分母会为0啊。
            # PunishnewEntropy += probfeat * calcShannonEnt(subDataSet)
        # 在划分数据的时候，有可能出现特征取同一个值，那么该特征的熵为0，
        # 同时信息增益也为0(类别变量划分前后一样，因为特征只有一个取值,如：民族（汉）)，
        # 0 / 0没有意义，可以跳过该特征。
        if featEntropy == 0:
           # value of the feature is all same,infoGain and iv all equal 0, skip the feature
           continue
        InfoGain = (baseEntropy - newEntropy) / featEntropy
        if InfoGain > bestInfoGain:
            bestInfoGain = InfoGain
            bestFeature = i
    return bestFeature


# 改写算法：自己推导的公式：Ent(play) = -sum(pi*log2pi) = -1/x * sum(xi*log2xi) + log2x
# 说明：X是属性outlook的字段sunny的样本个数，如：5，xi:sunny的分割数据子集对应的类别的样本数，如：1,2,2.
# 每个属性每个字段的分割数据子集个数不同，各类别对应的类别样本数也不同。
# 算法高效的技巧！：代码复用。改进算法：一从细节，二从思想。半自动：人工事先指定某些特征的权重。


def chooseBestFeatureToSplit1(dataSet):   # 选择出最好的特征，1.先用新思路实现
    numFeatures = len(dataSet[0]) - 1
    numEntries = len(dataSet)  # 分母
    # baseEntropy = 0.0
    baseEntropy = -1/numEntries * calcShannonEnt1(dataSet) + log(numEntries, 2)  # 类别熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 首先获取某一特征的所有可能取值集合
        # 从列表中创建集合是python获得列表中唯一元素值得最快方法
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # PunishnewEntropy = 0.0
        for value in uniqueVals:
            # featEntropy = 0.0
            subDataSet = splitDataSet(dataSet, i, value)
            X = len(subDataSet)
            probfeat = X / len(dataSet)  # 特征的概率
            # prob = float(X/numEntries)
            # calcShannonEnt(subDataSet) 需要重写！
            # featEntropy = calcShannonEnt1(subDataSet)
            featEntropy = -1/X * calcShannonEnt1(subDataSet) + log(X, 2)
            # print("ValsEntropy==0", value, featEntropy)    # 验证计算结果！
            newEntropy += featEntropy * probfeat
            # print("分母是： %f" %newEntropy)
            # 要惩罚具有较多特征取值的特征！信息熵归一化，分母会为0啊。
            # PunishnewEntropy += probfeat * calcShannonEnt(subDataSet)
        InfoGain = baseEntropy - newEntropy
        if InfoGain > bestInfoGain:
            bestInfoGain = InfoGain
            bestFeature = i
    return bestFeature

# C4.5未剪枝算法，首先要计算当前数据子集的属性分离信息，
# 即，spltRation(S,A) = --sum((Si/S)log2(Si/S)),Si是当前属性的字段的样本个数。
# 如，splitRate(S,outlook) = -(Num(sunny)/Num(S))*log2((Num(sunny)/Num(S))) +
# -(Num(rain)/Num(S))*log2((Num(rain)/Num(S))) + -(Num(overcast)/Num(S))*log2((Num(overcast)/Num(S)))


def featspltRation(dataSet, bestFeature):
    # bestFeature为当前层根节点，即划分数据子集的最佳特征索引，与label中一致！
    numEntries = len(dataSet)  # 分母
    ValsEntropy = 0.0
    featList = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featList)
    numVals = len(uniqueVals)
    countVals = []
    # s=['wsd','aws','edf']  s0=set(s) s0=={'aws', 'edf', 'wsd'},顺序改变了！但是，没关系，我要的是计算结果而已！
    for i in range(numVals):
        countVals.extend([featList.count(uniqueVals[i])])   # [int]为迭代对象。
        ValsEntropy += countVals[i] * log(countVals[i], 2)   # 用新公式计算
    ValsEntropy += -1/numEntries * ValsEntropy + log(numEntries, 2)
    return ValsEntropy


def chooseBestFeatureToSplit2(dataSet):   # 选择出最好的特征，新思路计算方法改成C4.5。
    numFeatures = len(dataSet[0]) - 1
    numEntries = len(dataSet)  # 分母
    baseEntropy = 0.0
    baseEntropy += -1/numEntries * calcShannonEnt1(dataSet) + log(numEntries, 2)  # 类别熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 首先获取某一特征的所有可能取值集合
        # 从列表中创建集合是python获得列表中唯一元素值得最快方法
        featList = [example[i] for example in dataSet]
        uniqueVals = list(set(featList))
        newEntropy = 0.0
        # 要不把属性分离信息在这里计算吧，因为需要的常量已经有了，无需再重复计算。
        ValsEntropy = 0.0  # 属性分离信息
        # PunishnewEntropy = 0.0
        # 需要改代码...
        numVals = len(uniqueVals)
        countVals = []
        # for value in uniqueVals:   # uniqueVals = set(featList) 顺序改变也没关系~
        for k in range(numVals):
            # featEntropy = 0.0
            countVals.extend([featList.count(uniqueVals[k])])  # [int]为迭代对象。
            # ValsEntropy += countVals[k] * log(countVals[k], 2)  # 用新公式计算
            # ValsEntropy += -1 / numEntries * ValsEntropy + log(numEntries, 2)
            # 为了将属性分离信息和选择最佳属性的2个循环合并，用另一种新公式计算。
            # 新公式：Ent(A)=-sum(Pi*log2Pi)=-sum((Xi/X)*log2(Xi/X))=-sum(Xi/X*(log2Xi-log2X))
            # ValsEntropy += countVals[k]/numFeatures * (log(countVals[k], 2) - log(numFeatures, 2))
            probfeat = countVals[k] / numEntries
            ValsEntropy -= probfeat * (log(countVals[k], 2) - log(numEntries, 2))
            # print("ValsEntropy==0", uniqueVals[k], ValsEntropy)
            subDataSet = splitDataSet(dataSet, i, uniqueVals[k])
            # probfeat = len(subDataSet)/len(dataSet)   # 特征的概率
            # X = len(subDataSet)
            # calcShannonEnt(subDataSet) 需要重写！
            featEntropy = -1/countVals[k] * calcShannonEnt1(subDataSet) + log(countVals[k], 2)
            # 得乘以概率。
            # probfeat = X / len(dataSet)  # 特征的概率
            newEntropy += featEntropy * probfeat
            # print("分母是： %f" %newEntropy)
            # 要惩罚具有较多特征取值的特征！信息熵归一化，分母会为0啊。
            # PunishnewEntropy += probfeat * calcShannonEnt(subDataSet)
        # 在划分数据的时候，有可能出现特征取同一个值，那么该特征的熵为0，
        # 同时信息增益也为0(类别变量划分前后一样，因为特征只有一个取值,如：民族（汉）)，
        # 0 / 0没有意义，可以跳过该特征。
        if ValsEntropy == 0:
            # value of the feature is all same,infoGain and iv all equal 0, skip the feature
            continue
        InfoGain = (baseEntropy - newEntropy)/ValsEntropy  # 除以分离信息
        if InfoGain > bestInfoGain:
            bestInfoGain = InfoGain
            bestFeature = i
    return bestFeature

# 3.1.3 递归构建决策树
# 终止条件：1、叶子节点的分类只有一个
# 2、所有属性都划分后，叶子节点分类仍不唯一，那么就采用多数投票方法。
# majorityCnt投票函数


def majorityCnt(classlist):   # 投票机制
    classCount = {}
    for vote in classlist:
        # vote，输出classCount[vote]（数量，否则为0）
        classCount[vote] = classCount.get(vote, 0) + 1
        # if vote not in classCount:
        #     classCount[vote] = 0
        # classCount[vote] += 1
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reversed=True)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1))
    return sortedClassCount

# 创建树的函数代码，肯定是递归，但是要保存子树，什么格式保存？嵌套形式保存，这样和递归结构一致！
# 为每一个类建造一颗决策树，建立N个类别的树，投票机制选出最佳类别。但是，这又和随机森林不同吧？有机会试试~


def createTree(dataSet, labels):   # labels是：与dataset样本特征顺序一致的特征列表，就是：决策树根节点。
    # 获得类别标签列表
    classList = [example[-1] for example in dataSet]
    # 全部属于同一类别，则停止划分，返回类别标签。所以，不会出现分母为0的！
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 否则，遍历完所有特征后多数投票机制选出类别标签！
    # 决策树是要使用完全部的特征，但是每个类别的相关特征是不同的，
    # 即某个特征对类的划分贡献度不同，所以停止条件也可以是数据集划分前后：信息熵的增加逐渐小于某个阈值。
    # 阈值 经验值取多少呢？（二八定律？）阈值之后就用多数投票机制选出最佳类别标签。
    # 决策树的评价函数可以改进，增加：特征冗余性评价，即识别分类能力相近的特征。
    # 但是，也可以用PCA事先去除分类能力低的特征。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 分类没有结束，就递归构造决策树。
    # bestFeat = chooseBestFeatureToSplit(dataSet)   # 原始方法原始C4.5
    # bestFeat = chooseBestFeatureToSplit1(dataSet)   # 测试！
    bestFeat = chooseBestFeatureToSplit2(dataSet)
    bestFeatLabel = labels[bestFeat]   # 获得划分当前数据集最好的特征，即当前层的根节点。
    # 字典变量myTree存储了树的所有信息，对于绘制树形图很重要。
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])  # 删除！！！交叉验证！注意！
    # 分别保存当前最好特征的各字段的数据子集，即子树。
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        # 防止每次调用时原始特征标签labels发生改变，保证某一特征所有取值遍历构造树前当前原始特征标签labels不改变
        #  因为递归调用createTree(dataSet, labels)，会发生 del(labels[bestFeat])，这样：
        #  Outlook的sunny、rain、cloud属性创建完子树后，就会删除outlook；再遍历Temperature时，
        #  根据再遍历Temperature属性创建子树时，labels和数据样本特征顺序不对应！就找不到Temperature的对应属性值
        # 和待测实例的对应属性值对应（labels中Temperature索引现在为0，而样本中是为2），就无法形成正确的决策树！
        subLabels = labels[:]
        # 获得最好特征的各字段对应的子树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)   # 这一步好棒！
    # print("createTree", type(myTree))
    return myTree


# 怎么在此基础上改进实现C4.5算法？
# 3.2 在python中使用Matplotlib注解绘制树形图
# 3.2.1 Matplotlib ...
# 3-8 使用决策树的分类函数


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]    # 获得第一个分类特征   !!!!
    secondDict = inputTree[firstStr]    # 获得第一个分类特征的子树
    featIndex = featLabels.index(firstStr)    # 获得第一个分类特征在featLabels的序号，featLabels和testVec特征顺序一致！
    classLabel = -1    # 可以作用到函数内，有其他例子。
    for key in secondDict.keys():
        # print("secondDict.keys() : %s" % (secondDict.keys()))
        # testVec中分割特征的取值 对应 子树的哪一个节点? 即 在同一层 走哪条路径？
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)   # 递归调用~
            else:
                classLabel = secondDict[key]
    # Local variable 'classLabel' might be referenced before assignment less... (Ctrl+F1)
    # This inspection warns about local variables referenced before assignment.
    return classLabel


# 使用十字交叉验证进行综合测试函数
from sklearn import cross_validation


def crossvalidation_trees(filename, testrate=0.3):
    # 程序写的是正确的，但是不能这样做，因为很有可能取不到全部的根节点，
    # 如特征名称列表为：['A', 'B', 'C', 'D'],但是，用训练集构造的树可能只用了'A','B'这两个列表名称~
    # 胡说，树的特征名称一定是featLabels的子集！
    fr = open(filename)
    dataSet = [inst.strip().split('\t') for inst in fr.readlines()]
    # 先： adult/yello数据集特征名称列表为：
    # featLabels = ['A', 'B', 'C', 'D']
    # 后： 数据集L-tic-tac-toe.data1.csv特征名称列表为：（类别：0-negative 1-positive）
    featLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I']
    errorRates = 0.0
    numTestVecs = int(len(dataSet) * testrate)
    for i in range(10):
        errorCount = 0.0
        # train_test_split从样本中随机按比例选取train data和test data。
        X_train, X_test = cross_validation.train_test_split(
            dataSet, test_size=testrate)  # random_state种子
        # 种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
        # 构造决策树
        subLabels = featLabels[:]
        myTree = createTree(X_train, subLabels)   # 注意：创建完树了，featLabels已经发生改变了，最终递归变为[]!!!
        # print("myTree 创建完后，featLabels 变为：", featLabels)   # 验证正确！找bug一从细节调试打印,二从原理细细捋！
        # print(type(myTree))    # 为什么是list？？txt文件格式不一样，分隔符必须是Tab!!!！要总结，不要重复犯错！
        for j in range(numTestVecs):
            classifierResult = classify(myTree, featLabels, X_test[j])
            if classifierResult != X_test[j][-1]:
                errorCount += 1.0
                # errorRate = errorCount / float(numTestVecs)
        errorRates += errorCount / float(numTestVecs)
        # print("the %d time's total error rate is: %f" % (i, errorRate))
    print("the 10 times classify0() average error rate is: %f" % (errorRates / 10.0))

# 3.3.2 使用算法：决策树的存储，避免每次分类都得重新构造决策树
# 3-9 使用pickle模块存储决策树


def storeTree(inputTree,filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)   # write() argument must be str, not bytes ...?
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


# 3-4 示例：使用决策树预测隐形眼镜类型

def glassTest(filename):
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    return lensesTree

# 剪枝法。再说~遇到问题，决不妥协！










