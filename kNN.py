# 2.1.1 准备：使用python导入数据
# 通用函数
# 测试算法前，必须确保将from os import listdir 写入文件的起始部分
# 代码主要功能是从os模块中导入函数lstdir,它可以列出给定目录的文件名！
# 为了数据可视化，我们只实现2维度特征
# import 语句用了才会高亮啊
from numpy import *
import numpy as np
import operator
from os import listdir


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 2.1.2 从文本文件中解析数据
# k-近邻 算法


def classify0(inX, dataSet, labels, k):
    # 改进方法1：将数据全部归一化到[0,1]后，因为x*x的导数为2x>=0,所以x*x在[0,1]上是单调递增的，
    # 所以x1*x1+x2*x2也是单调递增的，并且与x1+x2在[0,1]的单调性相同。
    dataSetSize = dataSet.shape[0]  # == group.shape[0]==4
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # ?
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  # 水平相加
    # distances = sqDistances**0.5
    # sortedDistIndicies = distances.argsort()
    sortedDistIndicies = sqDistances.argsort()
    # Indicies为index的复数形式，默认从小到大排序的数组元素索引号，与原始数据集中的一致
    classCount = {}   # 字典
    for i in range(k):
        # 这样labels数组才可以找到样本距离对应的标号
        voteIlabel = labels[sortedDistIndicies[i]]
        # voteIlabel在classCount里，输出classCount[voteIlabel]（数量，否则为0）
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    sortedclassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedclassCount[0][0]


def classify1(inX, dataSet, labels, k):   # inX, dataSet标准化到[0,1]！
    # 改进方法1：将数据全部归一化到[0,1]后，因为x*x的导数为2x>=0,所以x*x在[0,1]上是单调递增的，
    # 所以x1*x1+x2*x2也是单调递增的，并且与x1+x2在[0,1]的单调性相同。
    dataSetSize = dataSet.shape[0]  # == group.shape[0]==4
    # k近邻数目必须是1到样本个数之间的整数！
    if k < 0 or k > dataSetSize:
        print("k近邻数目必须是1到样本个数之间的整数！")
        return 0
    # ? 有正负啊，取绝对值。absolute!!!有时会慢。!错：因为值在[0,1]之间，所以都+1，就都>=0了！-0.1+1 =0.9，变大了。。
    diffMat = absolute(tile(inX, (dataSetSize, 1)) - dataSet)
    # inX += 0.5
    # diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # diffMat += 1.0   # 差值+1.0
    # sqDiffMat = diffMat**2
    # sqDistances = sqDiffMat.sum(axis=1)  # 水平相加
    sqDistances = diffMat.sum(axis=1)  # 水平相加
    # distances = sqDistances**0.5
    # sortedDistIndicies = distances.argsort()
    sortedDistIndicies = sqDistances.argsort()
    # Indicies为index的复数形式，默认从小到大排序的数组元素索引号，与原始数据集中的一致
    classCount = {}   # 字典
    for i in range(k):
        # 这样labels数组才可以找到样本距离对应的标号
        voteIlabel = labels[sortedDistIndicies[i]]
        # voteIlabel在classCount里，输出classCount[voteIlabel]（数量，否则为0）
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    sortedclassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedclassCount[0][0]

# 2.2 使用k-近邻算法改进约会网站的配对效果
# 2.2.1 准备数据：从文本文件中解析数据，转换为分类器可以处理的格式


def file2matrix(filename):
    # 打开文件.txt,本书格式。
    fr = open(filename)
    # 获取文件行数==样本数
    arrayOLines = fr.readlines()
    mfeature = len(arrayOLines[0].split()) - 1   # 最后一列是标号
    print("特征数为：%d" % mfeature)
    numberOfLines = len(arrayOLines)
    print("样本数为：%d" % numberOfLines)
    # 构造样本矩阵
    returnMat = zeros((numberOfLines, mfeature))   # 3是特征数，根据实际情况可以改变，可以改写代码！
    classLabelVector = []
    # 数据格式处理
    index = 0
    for line in arrayOLines:
        line = line.strip()   # 移除空格
        listFromLine = line.split('\t')   # 以换行符分离字符串
        returnMat[index, :] = listFromLine[0:mfeature]    # 0 1 2
        classLabelVector.append(int(listFromLine[-1]))   # 类别标号，所以数据准备一定要有意义并且符合格式。
        index += 1
    return returnMat, classLabelVector

# 处理csv格式数据
# my_matrix = numpy.loadtxt(open("c:\\1.csv","rb"),delimiter=",",skiprows=0)
# numpy.savetxt('new.csv', my_matrix, delimiter = ',')
# txt转为csv，注意：1.数据间英文逗号间隔 2.每行回车换行！ 之后，改后缀名为.csv


def file3matrix(filename):
    returnMat = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
    classLabelVector = returnMat[:, -1]
    returnMat = returnMat[:, 0:-1]
    return returnMat, classLabelVector

# 通过matplotlib绘图发现，对分类器有较大贡献度的特征组合为0/1，但是不知道各自的特征权重。先用PCA求出重要特征，
# 2.2.3 准备数据，归一化数值，消除不同数值范围上的差异，以防距离计算被大数值特征把控，而忽略了小数值的特征影响。
# 不归一化会影响特征权重的分配
# 归一化特征值


def autoNorm(dataSet):
    minVals = dataSet.min(0)   # 获得每列最小值，返回的是一个一维矩阵
    maxVals = dataSet.max(0)   # 获得每列最大值，返回的是一个一维矩阵
    ranges = maxVals - minVals
    # normDataSet = zeros(shape(dataSet))   # shape(dataSet) == （1000,3）
    m = dataSet.shape[0]   # m == 1000
    # 注意：特征值矩阵为1000x3,而minVals 和 maxVals 都为 1x3，所以要使用tile复制成与输入矩阵同样shape的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 要用优化形式,tile(ranges, (m, 1))为元素值为ranges（元组）的m行1列矩阵，所以这是特征值相除，不是矩阵相除。
    normDataSet /= tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 分类器针对约会网站的测试代码


def datingClassTest(filename):   # 没有十字交叉验证，不用。
    hoRatio = 0.10
    # datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # datingDataMat, datingLabels = file3matrix('iris.data1.csv')
    # datingDataMat, datingLabels = file3matrix('datingTestSet2 (3).csv')
    datingDataMat, datingLabels = file3matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 之前用十字交叉验证
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

# from timeit import repeat
# repeat测试可以防止程序运行测试时，后台其他程序运行的影响。repeat多次，取时间最小值即可。
# def func():
#     s = 0
#     for i in range(1000):
#         s += i
#
# #repeat和timeit用法相似，多了一个repeat参数，表示重复测试的次数(可以不写，默认值为3.)，返回值为一个时间的列表。
# t = repeat('func()', 'from __main__ import func', number=100, repeat=5)  单独测试
# print(t)
# print(min(t))
# 在Jupyter QtConsole中测试方法如下：
# from timeit import repeat
# func=kNN.cross_validation_kNN("datingTestSet2 (3).csv", testrate=0.3)
# t = repeat('func', 'from __main__ import func', number=1, repeat=5)   # t是一个时间列表！
# min(t)

from sklearn import cross_validation
#  aa = matrix([[1,1,1,1,1]])
# aa.partition()  concatenate():矩阵合并！！！


def cross_validation_kNN(filename, testrate=0.3):
    # 不对，应该把聚类结果写到新文件中！
    datingDataMat, datingLabels = file3matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    errorRates = 0.0
    numTestVecs = int(len(normMat) * testrate)
    # print("测试集大小为： ", numTestVecs)
    for i in range(10):
        errorCount = 0.0
        # train_test_split从样本中随机按比例选取train data和test data。
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            normMat, datingLabels, test_size=testrate)   # random_state种子
        # 种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
        # numTestVecs = len(X_test)
        # print("测试集大小为： ", numTestVecs)
        # print("the %d time's first test axample is: " % i, X_test[0])
        for j in range(numTestVecs):
            classifierResult = classify0(X_test[j, :],  X_train, y_train, 6)
            # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, y_test[j]))
            if classifierResult != y_test[j]:
                errorCount += 1.0
        # errorRate = errorCount / float(numTestVecs)
        errorRates += errorCount / float(numTestVecs)
        # print("the %d time's total error rate is: %f" % (i, errorRate))
    print("the 10 times classify0() average error rate is: %f" % (errorRates/10.0))


def cross_validation_kNN1(filename, testrate=0.3):
    datingDataMat, datingLabels = file3matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    errorRates = 0.0
    numTestVecs = int(len(normMat) * testrate)
    # print("测试集大小为： ", numTestVecs)
    for i in range(10):
        errorCount = 0.0
        # train_test_split从样本中随机按比例选取train data和test data。
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            normMat, datingLabels, test_size=testrate)  # random_state种子
        # 种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
        # numTestVecs = len(X_test)
        # print("测试集大小为： ", numTestVecs)
        # print("the %d time's first test axample is: " % i, X_test[0])
        for k in range(numTestVecs):
            classifierResult = classify1(X_test[k, :], X_train, y_train, 6)
            # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, y_test[j]))
            if classifierResult != y_test[k]:
                errorCount += 1.0
        # errorRate = errorCount / float(numTestVecs)
        errorRates += errorCount / float(numTestVecs)
        # print("the %d time's total error rate is: %f" % (i, errorRate))
    print("the 10 times classify1() average error rate is: %f" % (errorRates / 10.0))

# k-NN近邻算法改进：
# 1.找到2个全体局外点  2.计算2个全体局外点分别到待测试点的距离r1,r2
# 3.查找出[r1-d,r1+d]、[r2-d,r2+d]范围内的共有点，定位出待测试点的近邻所在“网格”区域
# 4. 数据归一化后，距离最远为N的0.5次幂(数据维度)，输入k,d则取dist(mins,maxs)/int(根号k)？

# 首先有一个函数：1、找2个局外点 2. 归一化，局外点也归一化 2.排序 ，计算绝对值和排序函数返回排序结果
# 分类函数： 1.分别计算待测样本和2个局外点的距离    2.在排序结果中定位待测点的位置
# 3.分别找出[r1-d,r1+d]、[r2-d,r2+d]内的共同点，d取N的0.5开方的1/10  4.统计出共同点最大类别，即是待测点的类别


def sortabsdist(normMat):
    # datingDataMat, datingLabels = file3matrix(filename)
    # 1 找2个基准点
    # datingDataMat[0] 和 minVals等的 shape 一样。
    # minVals = datingDataMat.min(0)  # 获得每列最小值，返回的是一个一维矩阵
    # maxVals = datingDataMat.max(0)  # 获得每列最大值，返回的是一个一维矩阵
    # ranges = maxVals - minVals
    # base1 = minVals 它们有共同引用，一改俱改！  base2 = minVals.copy() 创建副本引用就不相关了。
    # 2个基准点，base1， 一个是base2！
    # base2 = minVals.copy()
    # base2[-1] += ranges[-1]
    # 基于绝对值距离排序，先要归一化到[0,1]区间，2个基准点也要归一化。
    # datingDataMat.append(minVals) , 'numpy.ndarray' object has no attribute 'append'
    # normMat, ranges, minVals = autoNorm(datingDataMat)
    m, n = shape(normMat)
    # 值在[0,1]之间这么取才可以！
    base1 = zeros((1, n))  # 多维数组！！！
    base2 = base1.copy()
    # base2[-1] += 1   # [0 0 1],注意负值~
    # 改：base2[0][-1] += 1, 原因：base1 = zeros((1, n))为二维数组！
    base2[0][-1] += 1.0  # 与 base2 = base2[0][-1] + 1.0 不同。
    # base2 = (base2 - minVals) / ranges  # 数组对应元素相除
    # 计算到基准点base1的距离,排序
    diffMat11 = (normMat ** 2).sum(axis=1)  # 水平相加
    diffMat1 = diffMat11 ** 0.5   # 注意书写格式 diffMat1 = diffMat11 ** 0.5 与 diffMat1 **= 0.5 结果不同！！！
    base21 = tile(base2, (m, 1))    #
    diffMat21 = ((normMat - base21) ** 2).sum(axis=1)
    diffMat2 = diffMat21 ** 0.5
    # sortedDistIndicies2 = diffMat2.argsort()   # 处理负值。
    # bisect.bisect_left(qq1, qq2) 按照第一顺位元素排序，默认升序。 qq1=(0.87235069408852983,1000)
    # 字典没有len()
    sortdist1 = dict(zip(diffMat1, range(m)))
    sortdist2 = dict(zip(diffMat2, range(m)))
    # sortedDistIndicies1 = sortdist1.argsort()  # 'dict' object has no attribute 'argsort'
    # 按照距离排序，不能有缺失值。
    # 返回[(0, 'a'), (1, 'b'), (2, 'c')]格式，使用c[0]==(0, 'a'),c[0][1]=='a'
    sortdist1 = sorted(sortdist1.items(), key=lambda a: a[0])   # 结果为列表，bisect.bisect_left()用列表！
    sortdist2 = sorted(sortdist2.items(), key=lambda a: a[0])
    return sortdist1, sortdist2, m, n

import bisect


def sortabskNN(inX, sortdist1, sortdist2, m, n, datingLabels, s):
    # 距离等分的段数：是N的0.5开方/d  d=5必须放在后面！
    d = n ** 0.5 / s
    # print(d)
    # 1 分别计算base1、base2到inX的距离，inX标准化。
    distinX11 = (inX ** 2).sum(axis=0)   # ab=array([1,3,4]) 是数组：ab.sum(axis=0) == 8 ！
    distinX1 = distinX11 ** 0.5
    # zeros((1, n))== array([[ 0.,  0.,  0.]])  !!! base2 = zeros(n) == array([ 0.,  0.,  0.])
    base2 = zeros(n)   # 单列数组
    base2[-1] += 1.0
    distinX21 = ((inX - base2)**2).sum(axis=0)  # ?
    distinX2 = distinX21 ** 0.5
    # 2 搜索inX分别在sortdist1、sortdist2中的位置，二分查找方法。没有插入，不用删除。
    inXL1 = (distinX1 - d, m)
    inXR1 = (distinX1 + d, m + 1)
    inXL2 = (distinX2 - d, m)    # 有负值
    inXR2 = (distinX2 + d, m + 1)
    indexL1 = bisect.bisect_left(sortdist1, inXL1)
    indexR1 = bisect.bisect_right(sortdist1, inXR1)
    # print("indexL1 < indexR1 ", indexL1 <= indexR1)
    indexL2 = bisect.bisect_left(sortdist2, inXL2)
    indexR2 = bisect.bisect_right(sortdist2, inXR2)
    # print("indexL2 < indexR2 ", indexL2 <= indexR2)
    # tmp1 = sortdist1[indexL1, indexR1]
    # tmp2 = sortdist1[indexL1, indexR1]
    # rang1 = indexR1 - indexL1
    # rang2 = indexR2 - indexL2
    indexs = []
    # 只取到indexR1-1，下标从0的原因。list(range(3,7)) == [3,4,5,6]
    # 2层循环浪费时间啊~
    # 是我考虑不周！现在我不允许它们任何一方出现相等！for循环中出现任一range(0,0)
    # 都不会执行循环体，那么 indexs = []，程序出现异常！
    # 有这个条件判断后，可以跳过异常点，继续执行！示例如下！
    # 42 神马情况？ [0.37916256  0.47627156]
    # 0.09428090415820635
    # indexL1 < indexR1 True
    # indexL1 < indexR1 True
    # 划分整数d过大，请取小一些！小于当前整数d: 0.09428090415820635
    # 43 神马情况？ [0.91828551  0.88201179]
    # 0.09428090415820635
    # indexL1 <= indexR1 True
    # indexL2 <= indexR2 True
    # 44 神马情况？ [0.29999204  0.29898099]
    if indexL1 == indexR1 or indexL2 == indexR2:
        # print("划分整数s过大，请取小一些！小于当前整数s:", s)
        return 0
    # for i in range(0,0)是不执行的，
    # for i in range(0, 0):                             for i in range(0, 0):
    #     for i in range(0, 6)                              for i in range(0, 6)
    #         print(2)    # 执行无效！！！                        print(2)    # 执行无效！！！
    # sortindex1 = [for i in sortdist1[indexL1:indexR1]]
    # dict_values([9])   unorderable types: dict_values() < dict_values()
    # int() argument must be a string, a bytes-like object or a number, not 'dict_values'
    for i in range(indexL1, indexR1):   # 用二分查找。
        for j in range(indexL2, indexR2):
            # indexR2取不到!即：链表长度为900，最大为899，indexR2为900，循环只能取899！不会越界的！
            # if sortdist1[i][1] == sortdist2[j][1]:   # 有问题！
            if sortdist1[i][1] == sortdist2[j][1]:
                # sortdist1[i]是字典，字典取键值是sortdist1[i].values(),不是sortdist1[i][1]！
                # 而且，字典键值可以用==，!=  。
                indexs.extend([sortdist1[i][1]])  # a=[0,1,2] a.extend([4]) out: a==[0, 1, 2, 4]
    classCount = {}  # 字典
    k = len(indexs)
    for i in range(k):
        # 这样labels数组才可以找到样本距离对应的标号
        voteIlabel = datingLabels[indexs[i]]
        # voteIlabel在classCount里，输出classCount[voteIlabel]（数量，否则为0）
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # print("classCount[voteIlabel]: ", classCount[voteIlabel])
    # sortedclassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)   # import operator
    sortedclassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    # 遇到孤立点就扩大搜索间隔!这样就可以处理孤立点!但是循环要花费很多时间！
    if not sortedclassCount:
        # print("预测标号为：错误！")
        return 0
    else:
        # print("sortedclassCount[0][0] == """, sortedclassCount[0][0] == "")
        return sortedclassCount[0][0]

# 十字交叉验证算法
# 训练函数


def train_sortabskNN(filename, testrate=0.3):
    datingDataMat, datingLabels = file3matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        normMat, datingLabels, test_size=testrate)  # random_state种子
    sortdist1, sortdist2, m, n = sortabsdist(X_train)  # 太耗费时间，提前做。
    numTestVecs = len(X_test)
    return y_train, X_test, y_test, sortdist1, sortdist2, m, n, numTestVecs, X_train


def crosss_sortabskNN(y_train, X_test, y_test, sortdist1, sortdist2, m, n, numTestVecs, s):
    # datingDataMat, datingLabels = file3matrix(filename)
    # normMat, ranges, minVals = autoNorm(datingDataMat)
    errorRates = 0.0
    # print("测试集大小为： ", numTestVecs)
    # for i in range(10):
    errorCount = 0.0
        # train_test_split从样本中随机按比例选取train data和test data。
        # X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        #     normMat, datingLabels, test_size=testrate)  # random_state种子
        # 种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
        # numTestVecs = len(X_test)
        # print("测试集大小为： ", numTestVecs)
        # print("the %d time's first test axample is: " % i, X_test[0])
        # sortdist1, sortdist2, m, n = sortabsdist(X_train)   # 太耗费时间，提前做。
        # numTestVecs = len(X_test)
    for k in range(numTestVecs):
        classifierResult = sortabskNN(X_test[k, :], sortdist1, sortdist2, m, n, y_train, s)
        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, y_test[j]))
        # print(k, "神马情况？", X_test[k, :])
        if classifierResult != y_test[k]:
            errorCount += 1.0
        # print("出现问题的测试样本为： ", X_test[k, :])
        # errorRate = errorCount / float(numTestVecs)
    errorRates += errorCount / float(numTestVecs)
    # print("the %d time's total error rate is: %f" % (i, errorRate))
    print("the one time sortabskNN() average error rate is: %f" % errorRates)


# 先用库十字交叉验证，之后再自己写。
# from sklearn import cross_validation
# # model = RandomForestClassifier(n_estimators=100)
# #简单K层交叉验证，10层。
# cv = cross_validation.KFold(len(train), n_folds=10, indices=False)
# results = []
# # "Error_function" 可由你的分析所需的error function替代
# for traincv, testcv in cv:
#        probas = model.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
#        results.append( Error_function )
# print "Results: " + str( np.array(results).mean() )
# import numpy as np
# from sklearn import cross_validation
# >>> from sklearn import datasets
# >>> from sklearn import svm
# >>> iris = datasets.load_iris()
# >>> iris.data.shape, iris.target.shape
# ((150, 4), (150,))
# >>> X_train, X_test, y_train, y_test = cross_validation.train_test_split(
# ...     iris.data, iris.target, test_size=0.4, random_state=0)
# >>> X_train.shape, y_train.shape
# ((90, 4), (90,))
# >>> X_test.shape, y_test.shape
# ((60, 4), (60,))
# >>> clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# >>> clf.score(X_test, y_test)
# 0.96...
# _____________________________________________________________________________________________________________________
# 约会网站预测函数


def classifyperson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles eared per year?"))
    iceCream = float(input("liters of ice-cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([percentTats, ffMiles, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, datingDataMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


# 手写识别系统
# 2.3.1 准备数据：将图像转换为测试向量
# 将32x32的二进制图像转换为1x1024的向量，一行代表一个数字样本。


def img2vector(filename):
    returnVect = zeros((1, 1024))   #1行2014列
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        t = 32*i
        for j in range(32):
          returnVect[0, t+j] = int(lineStr[j])   # 必须是[0, t+j]
    return returnVect

# 测试算法：使用k-近邻算法识别手写数字
# 测试算法前，必须确保将from os import listdir 写入文件的起始部分
# 代码主要功能是从os模块中导入函数lstdir,它可以列出给定目录的文件名！
# 手写数字识别系统的测试代码,k决策树是进化版，计算开销小


def handwritngClassTest():
    hwLabels =[]
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)   # 一个样本一个测试
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(mTest)))

# k-近邻算法为什么不能由plt绘图直接看出来？非得要距离计算不可？数字识别可以依据轮廓啊！
# 有时候，直觉比逻辑更可靠！
# from sklearn.cluster import KMeans
# clusterer = KMeans(n_clusters=2,random_state=0).fit(reduced_data)

