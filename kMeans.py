# 利用K-均值聚类算法对未标注数据进行分组
# 聚类利用某种相似性度量标准，将样本进行自动归类。相似性度量可以用于样本、特征、特征组。向量内积度量依然可以使用
# “核技巧”，但是，在此之前需要把数据投影到高纬度空间，这样会有将样本间距离拉大甚至改变原有真实簇类结构的风险。相反，
#  可以降低到低维度，但是也有可能“过拟合”的风险，损失部分原有数据结构信息。一般聚类利用样本间相似度量，但是
# 每个类相关的特征子集是不同的，一般来说，好的特征子集使得类间分离度较大且类内分离度较小且不同类相关特征差异大，
# 同一类内特征相关性小，所以，从这种需求出发，一是可以改进聚类的目标优化函数，二是搭配其他算法使用。改进：
# “类依赖的加权”思想的应用。   迁移学习：基于“类关联”的类迁移思想的运用
# “类依赖加权”思想 + 决策树多分类 = 决策树并行多分类
# .......
# 聚类相似性度量对于离散型数据可以构造“核”，但要注意数据的有序性！到底使用哪种相似度方法取决于具体的应用。
# 数据集上K均值聚类算法的性能会受到所选距离计算方法的影响。还要注意：聚类有效性评价！
# “类依赖加权”思想的运用。
# 10-1 K-均值聚类算法支持函数
from numpy import *

# np.savetxt("reduced_data.csv",reduced_data)
# 处理csv格式数据
# my_matrix = numpy.loadtxt(open("c:\\1.csv","rb"),delimiter=",",skiprows=0)
# numpy.savetxt('new.csv', my_matrix, delimiter = ',')
# txt转为csv，注意：1.数据间英文逗号间隔 2.每行回车换行！ 之后，改后缀名为.csv


def file3matrix(filename):
    returnMat = loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
    # classLabelVector = returnMat[:, -1]
    # returnMat = returnMat[:, 0:-1]
    return mat(returnMat)

# 1 加载文件


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    arrOfLines = fr.readlines()
    mfeat = len(arrOfLines[0].split())
    # print("特征数目为： %d" % mfeat)
    for line in arrOfLines:
        # 从文件读入数据一定要将字符串转换成需要的数据类型！
        # curLine = line[:, -1].strip().split('\t')
        curLine = line.strip().split('\t')
        # 这样大数据读入数据效率太低了！
        flm = []
        for i in range(mfeat):
            flm.extend([float(curLine[i])])  # extend、append参数必须是可迭代的对象。
        dataMat.append(flm)
        # fltLine = map(float, curLine)   # 整行特征加入,但是map - map 不行！
        # dataMat.append(mat(curLine, float)) # shape(dataSet)=(80, 1, 2) 三维，每一维是一个2维矩阵!
        # 矩阵批处理
        # dataArr = [line.strip().split('\t') for line in fr.readlines()]
        # dataMat = [map(float, line) for line in dataArr]
    # return dataMat   # min(mat(dataMat)[:,0])
    return mat(dataMat)  # 这样其他函数就不用再转换成mat了。


# 2 计算欧式距离
def distEclud(vectA, vectB):
    # return sqrt(sum((vectA-vectB)**2))
    return sqrt(sum(power(vectA - vectB, 2)))

# 3 初始化类中心


def randCent(dataSet, k):
    # 怎么样更好地选择有代表性的K个点？涉及算法的收敛速度和计算效率！
    # 由于不知道数据的真实分布结构究竟是怎样的，从结构风险最小化来分析，可以依据样本各个特征取值进行K个均匀分段！
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    # 一个一个特征进行处理
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)   # map - map 不行！
        # K个中心第j个特征用矩阵同时批处理  random.rand(k, 1)返回K行1列数组！元素值在(0,1)之间的小数
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


# 4 K-均值决算法
# 函数接口设计很人性化，便于维护修改。


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    # 分配矩阵：[簇序号，最小簇中心距离平方] 最小簇中心距离平方：用于簇类质量评价
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    cluserChanged = True
    while cluserChanged:
        cluserChanged = False
        # 为每一个样本分配簇中心
        for i in range(m):
            minDist = inf
            minIndex = -1
            # 找出离样本最近的簇中心
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])   # 矩阵向量
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 填写分配矩阵
            if clusterAssment[i, 0] != minIndex:
                cluserChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        # print(centroids)
        # 重新计算簇均值
        for cent in range(k):
            # nonzero(clusterAssment[:, 0].A == cent)[0] 返回clusterAssment中
            # 所有行第一列等于cent的元素值的行号(样本序号)！
            # 版本：a1=np.nonzero([1==2]) a2=np.nonzero(1==2) 结果都是：(array([], dtype=int64),)，但是：a1 != a2.
            # 提示：iterrable对象才行。【1】可以。
            # type(ptsInClust):numpy.matrixlib.defmatrix.matrix,所以距离计算可以用矩阵批处理
            # 得写一个矩阵批处理的欧拉距离计算公式
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    # 簇中心收敛后，返回簇中心矩阵、分配矩阵
    return centroids, clusterAssment


# 如何评价聚类的有效性？
# K均值算法收敛但是聚类效果较差的原因是：聚类收敛到局部最小值而非全局最小值！
# 这很有可能是因为：簇中心随机初始化 造成的！改进：启发式分配簇中心！！！
# 一种用于度量聚类效果的指标是SSE(误差平方和)，一种肯定可以降低SSE的方法是：增加簇类数目，极端时SSE可为0！
# 但是，目标需求是：簇类数目不改变时，降低SSE。
# 量化方法：1 合并最近的质心  2 合并2个使得SSE增幅最小的质心。(在所有可能的簇上两两穷举，再计算总的SSE)

# 10-3 二分K-均值算法
# 一种：每次将每个簇一分为二，选择能够(SSE前-SSE后)最大程度减小SSE的簇分类。
# 二种：选择SSE最大的簇进行划分。
# 思考：能够用决策树基于信息熵来划分簇吗？设想：第一层分类节点数==簇数目K,依据每个节点将簇划分，
# 虽然可能造成一个样本被分到多类的情况，但是可以选择最大簇作为其类别。而且，在一个类混乱度达到较小情形时，
# 去掉某个点只会使该簇混乱度减小。复杂度:K*N每个特征又可以取多个值...，复杂度高啦，所以随机森林好啊！
# 信息熵计算与欧几里得计算哪个麻烦？

def biKmeans(dataSet, k, distMeas=distEclud):
    # dataSet最后一列必须是类别标号！
    m = shape(dataSet)[0]
    # 聚类数目范围必须在2到m-1之间的整数！
    if k == 1 or k == m:
        print("聚类数目范围必须为2到(样本个数m)-1之间的整数！")
        return 0
    clusterAssment = mat(zeros((m, 2)))
    # [-0.10361321250000004, 0.05430119999999998]  无[0]:[[-0.10361321250000004, 0.05430119999999998]]
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    while len(centList) < k:
        lowerSSE = inf  # Inf = inf = infty = Infinity = PINF
        for i in range(len(centList)):
            # index 2 is out of bounds for axis 1 with size 2，reduced_data1，k=4!出错！
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0]]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, i].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if sseSplit + sseNotSplit < lowerSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowerSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # print("the bestCentToSplit is: ", bestCentToSplit)
        # print("the len of bestClustAss is: ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment


# from sklearn.metrics import silhouette_score
# score = silhouette_score(dataSet, clustAssing2[0])
# samples_preds = clusterer.predict(pca_samples)

# 计算评价聚类质量的评价标准：轮廓系数
# silhouette系数是对聚类结果有效性的解释和验证，有Peter J.Rousseeuw与1986年提出。
# 1.计算样本i到同簇其他样本的平均距离ai。ai越小，说明样本i越应该被聚类到该簇。将ai称为样本i的类内不相似度。
# 簇C中所有样本的ai的均值称为簇C的簇不相似度。
# 2.计算样本i到其他某簇Cj的所有样本的平均距离bij，称为样本i与簇Cj的不相似度。
# 定义为样本i的簇间不相似度，bi=min{b1,..,bik}.bi越大，说明样本i越不属于其他簇。
# 3.根据样本i的簇内不相似度ai和簇间不相似度bi,定义样本i的轮廓系数：s(i)=(bi-ai)/max(bi,ai)
# 分段讨论：(1)ai<bi,s(i)=1 - ai/bi ,(2)ai==bi,s(i)=0, (3)ai>bi,s(i)=bi/ai - 1
# si 接近1，说明样本i越应该分配到当前簇；si 接近-1，说明样本i越应该分配到其他簇； si 接近0，说明样本i在边界上。
# 所有样本的si均值是平均轮廓系数，是对聚类质量评价的简单度量标准。


def calcSilhouette(dataSet, clusterAssment):
    # 需要输出时间，但是time有其他后台程序干扰，时间、系数只能分开讨论了。
    # k 做参数不合适，因为聚类结果已经定了。
    # clusterAssment[:, 0] 是矩阵，我需要矩阵元素去重
    k = len(set(array(clusterAssment[:, 0])[:, 0]))
    m = shape(dataSet)[0]
    # 重新计算簇均值
    si = 0.0
    for i in range(m):
        # 'range' object doesn't support item deletion
        ks = list(range(k))
        # ki = ks.index(clusterAssment[i, 0])
        # nonzero(clusterAssment[:, 0].A == cent)[0] 返回clusterAssment中
        # 所有行第一列等于cent的元素值的行号(样本序号)！
        # 版本：a1=np.nonzero([1==2]) a2=np.nonzero(1==2) 结果都是：(array([], dtype=int64),)，但是：a1 != a2.
        # 提示：iterrable对象才行。【1】可以。
        # type(ptsInClust):numpy.matrixlib.defmatrix.matrix,所以距离计算可以用矩阵批处理
        # 得写一个矩阵批处理的欧拉距离计算公式
        # nonzero(clusterAssment[:, 0].A == clusterAssment[i, 0])【这是对的！】 与
        # nonzero(【clusterAssment[:, 0].A == clusterAssment[i, 0]】)结果不同，这是错的！
        # 计算ai
        ptsInClusta = dataSet[nonzero(clusterAssment[:, 0].A == clusterAssment[i, 0])[0]]
        # nonzero(clusterAssment[:, 0].A == clusterAssment[i, 0] 返回的结果包括 i!
        # 所以这里ai计算除以|ck| !!!
        n = shape(ptsInClusta)[0]
        # tile（A，reps）:A:array_like!
        diffMat = tile(dataSet[i], (n, 1)) - ptsInClusta  #
        # array**2和matrix**2 不一样！数组是每个元素，矩阵是矩阵乘法。数组和矩阵引用方式不同而已。
        # 很多情况下，矩阵与多维数组可以换用。个人感觉，np.ndarray很好用！
        # 这时候：数组更方便array**2
        sqDiffMat = array(diffMat) ** 2
        sqDistances = sqDiffMat.sum(axis=1)  # 水平相加
        distances = sqDistances ** 0.5
        ai = average(distances)  # 数组
        # 计算bi
        bi = inf   # 无穷大
        ks.remove(clusterAssment[i, 0])
        for ki in ks:
            ptsInClustb = dataSet[nonzero(clusterAssment[:, 0].A == ki)[0]]
            n = shape(ptsInClustb)[0]
            diffMat = tile(dataSet[i], (n, 1)) - ptsInClustb  #
            sqDiffMat = array(diffMat) ** 2
            sqDistances = sqDiffMat.sum(axis=1)  # 水平相加
            distances = sqDistances ** 0.5
            # average(distances, axis=0) 是matrix([[1.]])形式！
            bij = average(distances)  # 数组
            #  找出最小bi
            if bij < bi:
                bi = bij
        # 计算si
        si += (bi - ai) / max(bi, ai)
    print("该次聚类的平均轮廓系数为：", si/m)
    return si/m

# aa = matrix([[1,1,1,1,1]])
# aa.partition()  concatenate():矩阵合并！！！
# 写一个函数，将聚类结果的dataMat、label写到新文件中！
# from numpy import *  import numpy as np


def writetofile(dataMat, labels, filename):
    dataSet = concatenate((dataMat, labels), axis=1)
    return savetxt(filename, dataSet, delimiter=',')
# 测试：file1 = writetofile(dataMat2, clusterAssment[:,0]) 可以的，
# 但是矩阵数据直接写进csv文件，用excel打开后矩阵同一行的数据全部都在一个表格单元中！怎么分开数据？delimiter=','是关键！

