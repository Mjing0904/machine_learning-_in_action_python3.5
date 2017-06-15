# 13 利用PCA来简化数据
# PCA首先进行坐标系转换，然后在新的坐标系中选择具有较大方差的维度作为主要特征候选。
# 例如，选取的第一个新坐标轴是原始数据中方差最大的方向。然后再选择与第一个新坐标轴正交的次大方差的第二个坐标轴。
# 对比决策树：每一个根节点的选择都是在当前分类的数据情况下选择信息增益率最大的维度，该维度是原始数据特征的某一个特征。
# PCA选择的新坐标轴不一定是原始特征的子集！
# 那么，可以用PCA处理后选出新特征，再用决策树分类吗？能否为每个类选择出最合适的特征子集呢？？？
# PCA降维处理后的信息损失评估？有什么注意事项吗？能否增加某些惩罚项防止数据信息损失较大？
# 2维PCA可以可视化显示。但是，多维数据怎么观测在哪一个维度上分类效果更好？怎么利用PCA为每一个类找出最优特征子集？
# 基于PCA的特性，特征权重也可以方差计算。一个类一个类逐个特征地找？
# 决策树每次分类都是只根据一个特征根节点来对整个数据集进行划分。在同一层次，能否同时根据多个特征进行不同类的划分？
# 就是说，每一层同时进行多个类别划分，每个特征都是对应各个类别最有贡献度的特征,这样比随机森林复杂度低、准确率高吧？
# 上述是：决策树并多分类 算法设计主要思想。
# K-NN算法每次都要进行全数据集的距离计算，这样算法效率太低了！能否事先用聚类对数据集进行划分，
# 然后再用监督学习算法训练模型？怎么改进K-NN呢？计算各个类中心，离谁近就是哪家的呗？这是 K均值聚类？
# K-NN算法每次都是进行距离度量，修改样本相似性度量标准，如：向量内积。 既然用到向量内积的话，那么也可以进行高维投影，
# 然后使用“核技巧”计算样本相似性？这样就是 K-NN+核 啦。也可以使用信息熵度量。
# 我发现算法的灵活使用主要就是算法思想的灵活组合嘛。
# SVM多分类器~~~

# 13-1 PCA算法:numpy中模块linalg的eig()方法，求解特征向量和特征值。

from numpy import *

def loadDataSet(filename):
    fr = open(filename)
    stringArr = [line.strip().split('\t') for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]  # 矩阵批处理
    return datArr


def pca(dataMat, topNFeat = 9999):   # 输入要灵活...
    # 每列计算均值，得到1XN矩阵
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    #
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNFeat + 1):-1]
    reEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * reEigVects
    reconMat = (lowDDataMat * reEigVects.T) + meanVals
    return lowDDataMat, reconMat









