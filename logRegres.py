# 5-1 Logistic回归梯度上升优化算法
from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    arrayOLines = fr.readlines()
    # 特征数由文件任意给定，不是固定的。修改成功。
    # 关键是细节处理：在QTconcole下自己试验就摸索出规律了。坚持就是胜利！还好没放弃！
    # 你要带着脑子思考提出尽可能多的问题！多处理修改细节！加油！
    mfeature = len(arrayOLines[0].split()) - 1  # 最后一列是标号
    for line in arrayOLines:
        lineArr = line.strip().split()
        # 最后一列是类别，没有包括。(改写了原书代码)
        flm = [1.0]
        for i in range(mfeature):
            flm.extend([float(lineArr[i])])
        dataMat.append(flm)
        labelMat.append(int(lineArr[-1]))   # error = (labelMat - h) 少了int就不能计算！
    # a, b = shape(dataMat)
    # print(a, b)
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 该梯度上升算法有不足：
# 1 算法的最大迭代次数人为指定  2 算法迭代停止标准还可以是：每次dataMat.transpose() * (labelMat - h)小于某个阈值
# 3 步长的经验值是多少？阈值又该设为多少？


def gradAscent(dataMatIn, classlabels):
    # 转换为numpy矩阵数据类型,计算量大为：特征数*样本数
    # 原因：每次更新回归系数时都需要遍历整个数据集：批处理
    dataMat = mat(dataMatIn)
    labelMat = mat(classlabels).transpose()  # 行向量，使用下标
    m, n = shape(dataMat)
    # numpy.ndarray
    weights = ones((n, 1))
    alpha = 0.001
    # 设置最大迭代次数或者阈值
    maxCycles = 500
    for k in range(maxCycles):
        # 以下是numpy矩阵运算！！！h是一个列向量！
        # 计算整个数据集的梯度：推导自己推。
        # 计算量大为：特征数*样本数 not dictClass = sigmoid(dataMat[k] * weights)
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights += alpha * dataMat.transpose() * error  # dataMat.transpose():nxm error:mx1
    return weights


# 5.2.3 分析数据：画出决策边界
# 5-2 画出数据集和Logistic回归最佳拟合直线的函数

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    # 'numpy.ndarray'object has no attribute 'getA'
    # Return self as an ndarray object.Equivalent to np.asarray(self)
    # weights = wei  也不用 weights.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]   # 只能画2维特征，此处设w0x0+w1x1+w2x2=0
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


# 5.2.4 训练算法：随机梯度上升
# 一次只用一个样本点来更新回归系数。
# 因为可以在新样本到来时对分类器进行增量式更新，又称为：在线学习算法。
# 5-3 随机梯度上升算法:每次运行都有可能获得不同的结果。


def stocGradAscent0(dataMatrix, classlabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   # not ones((n, 1))
    for i in range(m):
        # 这一步可以改进：运用 启发式策略！
        h = sigmoid(sum(dataMatrix[i] * weights))  # weights：1xn dataMatrix[i]:1xn
        error = classlabels[i] - h
        # 如果结果不收敛咋办？ 但是这里是只有一个最优解，所以不会出现抖动现象，但是仍要注意。
        weights += alpha * error * dataMatrix[i]   # 每个维度特征一次数组更新
    return weights


# 一个判断优化算法优劣的可靠方法就是看它是否收敛。
# 图5-6展示运用随机梯度上升算法：回归系数在经历较大的迭代次数后，经过了大的波动停止后，还有一些小的周期性波动。
# 不难理解，产生这种现象的根本原因是：存在一些不能正确分类的样本点（数据集并非线性可分）。
# 算法改进： 1 避免在某个值附近来回振动   2 算法收敛速度加快。
# 5-4 改进的随机梯度上升算法


def stocGradAscent1(dataMatrix, classlabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)   # not ones((n, 1))
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # alpha每次迭代都会减小，但永远不会是0，因为还有一个常数项0.01.
            # 必须这样做原因为：保证在多次迭代后新数据还会有一定的影响。
            # 如果是动态变化的，那么可以加大常数项值，确保新样本有更大的回归值。
            # 另一点值得注意的是：在降低alpha的函数中，alpha每次减少1/(j+i),j为迭代次数，i为样本数。
            # 当j<<max(i)时，alpha就不是严格下降的。避免参数的严格下降也是见于模拟退火等其他优化算法中。
            alpha = 4/(1.0+j+i)+0.01
            # 随机选择样本数，可以减少周期性的波动
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # weights：1xn dataMatrix[i]:1xn
            error = classlabels[randIndex] - h
            weights += alpha * error * dataMatrix[randIndex]   # 每个维度特征一次数组更新
            delete(dataIndex, randIndex)
            # del(dataIndex[randIndex])
    return weights


# reload(logRegres)
# dataArr,labelMat = logRegres.loadDataSet()
# weights=logRegres.gradAscent(dataArr,labelMat)
# weights=logRegres.stocGradAscent0(array(dataArr),labelMat)

# 画图时，stocGradAscent0算出的weights，被plotBestFit()调用出现index 1 is out of bounds for axis 0 with size 1 ???
# 解释：传入参数直接用weights，不用weights.getA(),也不要用mat(weights).getA(),
# 因为传入的参数就是需要的np.ndarray形式，无需改变。
# logRegres.plotBestFit(weights)

# 5.3 示例：从疝气病症预测病马的死亡率
# 5.3.1 准备数据：处理数据中的缺失值
# 1 使用可用特征的均值来填补缺失值  2 使用特殊值来填补缺失值，如-1   3 忽略有缺失值的样本
# 4 使用相似样本的均值填补缺失值  5 使用另外的机器学习算法预测缺失值
# 数据预处理：1 填补缺失值
# 因为1 numpy矩阵不允许有缺失值
# 因为2 实数0恰好适用于logistic回归。直觉是：我们需要的是一个：在更新时不会影响系数的值。
# weights += alpha*dataMatrix[randIndex]。如果dataMatrix[randIndex]的某个特征为0，那么该特征的系数将不做更新。
# 而且，sigmoid(0)=0.5，对于结果的预测不具有任何的倾向性，因此也不会对误差项造成影响。
# 所以，实数0做缺失值，既保留现有数据，也满足该特征值缺失的“特殊性”。

# 数据预处理： 2 丢弃 类别缺失样本
# 说明：logistic回归可以丢弃，knn不适合丢弃。查文献！

# 5.3.2 测试算法：用logistc回归进行分类
# 5-5 logistic回归分类函数


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():   # 此数据集21个特征，最后一列为类别。
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():   # 行处理
        # trainingSet.append(line.strip().split())
        currLine = line.strip().split('\t')
        lineArr = []
        # 加载样本...
        for i in range(21):
            lineArr.append(float(currLine[i]))  # 用extend更好，是[[1,2,...,21],[1,2,..,21]]形式。
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeghts = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeghts)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount)/numTestVec
    print("The errorRate of this test is: %f" %errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("After %d iterations ,the average errorRate is: %f" % (numTests, errorSum/float(numTests)))













