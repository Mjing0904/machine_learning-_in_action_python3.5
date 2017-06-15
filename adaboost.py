# 7 利用Adaboost元算法提高分类性能


def loadSimpData():
    dataMat = matrix([1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

# 7-1 单层决策树生成函数
def stump
