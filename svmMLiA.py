# dataMatIn=[[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]            a=[[1,2,3,4]]
# 列表
# dataMatIn                                                       a
# [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]        [[1, 2, 3, 4]]
# 矩阵
# mat(dataMatIn)                                                  mat(a)                         mat(a).transpose()
# matrix([[1, 2, 3, 4],                                           matrix([[1, 2, 3, 4]])         matrix([[1],
#         [2, 3, 4, 5],                                                                                  [2],
#         [3, 4, 5, 6],                                                                                  [3],
#         [4, 5, 6, 7]])                                                                                 [4]])


# SVM 知识回忆：
# w* = a1z1x1+a2z2x2+...+amzmxm (m为样本数，n为特征数)
# 预测新样本公式： w*x+b = (a1z1x1+a2z2x2+...+amzmxm)*x+b = a1z1(x1*x)+a2z2(x2*x)+...+amzm(xm*x)+b
# b*的软正则化情况下求解看Platt论文
# 6-1 SMO算法中的辅助函数
from numpy import *


def loadDataSet(filename):
    # 返回的是列表！！！ dataMatIn=[[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]
    dataMat = []
    labelMat = []
    fr = open(filename)
    arrOfLines = fr.readlines()
    mfeat = len(arrOfLines[0].split()) - 1
    for line in arrOfLines:
        # lineArr = line.strip().split()
        lanArr = line.strip().split('\t')
        flm = []
        # 特征数自适应
        for i in range(mfeat):
            flm.extend([float(lanArr[i])])   # extend、append参数必须是可迭代的对象。
        dataMat.append(flm)
        labelMat.append(float(lanArr[-1]))   # 少了int就不能计算！
    return dataMat, labelMat


def selectJrand(i, m):
    # 选择m范围内不等于i的J
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    # H:high L:lower 动态求解二次最优解的边界处理，详情见推导。
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 6-2 简化版SMO算法


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 矩阵批处理
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    # 数组：zeros((m, 1))   矩阵： mat(zeros((m, 1)))
    alphas = mat(zeros((m, 1)))
    # iter ：Shadows built-in name 'iter'
    iter1 = 0
    while iter1 < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # multiply():矩阵对应位置元素相乘。 .T :转置  *:线性代数矩阵相乘(相乘相加)
            # 预测新样本公式： w*x+b = (a1z1x1+a2z2x2+...+amzmxm)*x+b = a1z1(x1*x)+a2z2(x2*x)+...+amzm(xm*x)+b
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])   # 误差
            # 应该满足的KKT条件为：
            #  1. 函数间隔 > 1,即样本远离边界线，则alphas[i]==0
            #  2. 函数间隔是1，即样本在边界线上，则 0 < alphas[i] < C
            #  3. 函数间隔小于1，即样本在2条边界线之间，则 alphas[i] == C
            #  不满足KKT条件的优先调整：
            #  1. 函数间隔 > 1,即样本远离边界线，即labelMat[i] * Ei > toler 但是 alphas[i] > 0
            #  2. 函数间隔 < 1，即样本在2条边界线之间且不在误差范围内，即labelMat[i] * Ei < -toler，但是 alphas[i] < C
            #  3. 函数间隔是1，即样本在边界线上，但是 alphas[i] ==0 或 alphas[i] == C
            #  为什么第3种情况不考虑？
            if labelMat[i] * Ei < -toler and alphas[i] < C or labelMat[i] * Ei > toler and alphas[i] > 0:
                # 随机选择alpha j,此时已固定alpha i,所以，接下来是优化关于alpha j的函数。
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 为更新alphas[j]后更新alphas[i]、b做准备、比较值是否改变。用copy()是为了建立不同的引用！
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 计算alphas[j]上下界
                if labelMat[i] != labelMat[j]:
                    # 此时 zi != zj,此时由 aizi+ajzj=$,分析得：
                    # zi=1,zj=-1，ai-aj=$,目标优化函数看做关于aj的二次函数，则：Max(aj)=C-$,Min(aj)=-$。
                    # zi=-1,zj=1，-ai+aj=$',目标优化函数看做关于aj的二次函数，则：Max(aj)=$'+C,Min(aj)=$'。
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    # labelMat[i] == labelMat[j] 即 zi == zj,此时由 aizi+ajzj=$,分析得：
                    # zi==zj==1 时，ai+aj=$,目标优化函数看做关于aj的二次函数，则：Max(aj)=$,Min(aj)=$-C。
                    # zi==zj==-1时，ai+aj=$',目标优化函数看做关于aj的二次函数，则：Max(aj)=$',Min(aj)=$'-C。
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    # 由 aizi+aizj=$,怎么会出现这种情形？
                    print("L == H")
                    continue
                # 计算分母：2K12-K11-K22 == -(K11+K22-2K12) < 0
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                # 如果分母是K11+K22-2K12计算的，就是+=。
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                # 再结合alphas[j]的范围，给出正确的alphas[j]更新结果。
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 观察每次更新的幅度
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 更新alphas[i]
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 更新b，b的公式我没推导。
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # 根据情况选择合适的b值
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" % (iter1, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter1 += 1
        else:
            iter1 = 0
        print("iteration number: %d" % iter1)
    return b, alphas


# 6.4 利用完整Platt SMO算法加速优化
# 6-3 完整版Platt SMO 算法支持函数


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):   # __init__ !!!
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))   # 误差缓存


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    EK = fXk - float(oS.labelMat[k])
    return EK


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 6-4 完整Platt SMO 算法中的优化例程


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if oS.labelMat[i]*Ei < -oS.tol and oS.alphas[i] < oS.C or oS.labelMat[i]*Ei > oS.tol and oS.alphas[i] > 0:
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            # 此时 zi != zj,此时由 aizi+ajzj=$,分析得：
            # zi=1,zj=-1，ai-aj=$,目标优化函数看做关于aj的二次函数，则：Max(aj)=C-$,Min(aj)=-$。
            # zi=-1,zj=1，-ai+aj=$,目标优化函数看做关于aj的二次函数，则：Max(aj)=$+C,Min(aj)=$。
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            # labelMat[i] == labelMat[j] 即 zi == zj,此时由 aizi+ajzj=$,分析得：
            # zi==zj==1 时，ai+aj=$,目标优化函数看做关于aj的二次函数，则：Max(aj)=$,Min(aj)=$-C。
            # zi==zj==-1时，ai+aj=$,目标优化函数看做关于aj的二次函数，则：Max(aj)=$,Min(aj)=$'-C。
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # 由 aizi+aizj=$,怎么会出现这种情形？
            print("L == H")
            return 0
            #
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# 6-5 完整版Platt SMO的外循环代码


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and alphaPairsChanged > 0 or entireSet:
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print("fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number : %d" % iter)
    return oS.b, oS.alphas






















