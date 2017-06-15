# 朴素贝叶斯基于2点基本假设：
# 1 特征条件独立   2 特征是平等的，无区别。

# 4.5 使用python进行分类
# 4.5.1 准备数据： 从文本中构建词向量
# 4-1 词表到向量的转换函数
from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'stupid', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]   # 1 代表侮辱性文字,人工标注？
    return postingList, classVec



def createVocaList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet |= set(document)   # 无重复元素，但是之前应该去除高频词！
    return list(vocabSet)    # 为什么返回列表，引用方便。
 # a=[1,2,3,4]  b=a[0::3] b改，a不改。


def setOfWords2Vec(vocabList, inputSet):
    # 每篇文档的词向量维度相同，没有考虑不同文档的长度及词出现的次数
    returnVec = [0] * len(vocabList)   # 文档向量的特点：高维、稀疏，设置0是合理的。
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  # 词的对应序号为1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 4.5.2 训练算法：从词向量计算概率
# 4-2 朴素贝叶斯分类器训练函数： 计算条件概率p(wi|c) 、先验概率p(c)
# for postinDoc in listOPosts:   #for循环填充词向量列表
#     trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))


def trainNB0(trainMatrix, trainCategory):   # 改写成多分类。
    numTrainDocs = len(trainMatrix)   # 文档数
    # 二分类： 1-pAbusive ，多分类怎么改写代码？ K分类向量，计算k-1维。尝试考虑不同文档的长度及词出现的次数。
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    numWords = len(trainMatrix[0])
    p0Num = ones(numWords)   # 矩阵的便利，批处理  p0Num = zeros(numWords)
    p1Num = ones(numWords)   # 矩阵的便利，批处理  p1Num = zeros(numWords)
    p0Denom = 2.0   # p0Denom = 0.0 多分类：p0Denom = k.0
    p1Denom = 2.0   # p1Denom = 0.0
    for i in range(numTrainDocs):   # not trainMatrix  'int' object is not iterable
        if trainCategory[i] == 1:
            # 问题：如果有新的特征词，并且该新特征词从未在训练文档中出现，那么该文档的分类概率就一直是0.
            # 所以，对于有未在训练文档中出现的特征词的待分类文档，就判定该文档无法分类是极不合理的。
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])   # 每篇文档词出现的次数未考虑，但是所有文档整体考虑了。
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)   # p1Vect = p1Num / p1Denom
    p0Vect = log(p0Num / p0Denom)   # p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


def trainNB1(trainMatrix, trainCategory):   # 多分类。
    numTrainDocs = len(trainMatrix)   # 文档数
    # 首先知道类别和类别数目
    slCategory = list(set(trainCategory))   # 类别列表['f','a','c']
    cates = len(slCategory)
    cates1 = cates - 1
    PCategory = zeros(cates)   # 类别概率列表array([ 0.,  0.,  0.])
    for category in trainCategory:
        for cate in range(cates1):
            if category == slCategory[cate]:
                PCategory[cate] += 1
    PCategory /= numTrainDocs   # k-1个类别概率知道了，数量级很大。
    PCategory[-1] = 1.0 - PCategory.sum(axis=0)   # 纵轴
    # 计算各属性在各个类别中的概率
    numWords = len(trainMatrix[0])
    pfeatNum = ones((cates, numWords))
    # 各类别的初始化分母
    pfeatDenom = zeros(cates) + float(cates)   # 数组加常数。
    # 统计各类别中各属性的频数
    for i in range(numTrainDocs):   # not trainMatrix  'int' object is not iterable
        for j in range(cates):
            if trainCategory[i] == slCategory[j]:
                pfeatNum[j] += trainMatrix[i]
                pfeatDenom[j] += sum(trainMatrix[i])
    # 取对数计算概率
    pVect = zeros((cates, numWords))
    for j in range(cates):
        pVect[j] = log(pfeatNum[j] / pfeatDenom[j])   # 数组除以常数
    return pVect, PCategory, slCategory

# 测试算法： 根据现实情况修改分类器：
# 1、拉普拉斯平滑：
# 解决问题：如果有新的特征词，并且该新特征词从未在训练文档中出现，那么该文档的分类概率就一直是0.
# p0Num = ones(numWords)   # 矩阵的便利，批处理
# p1Num = ones(numWords)   # 矩阵的便利，批处理
# p0Denom = 2.0
# p1Denom = 2.0
# 下溢出：当计算条件概率乘积时，可能因为条件概率是很小的小数，造成程序下溢或者最终得到结果为0.
# 解法1：对乘积取自然对数。好处1.ln(a*b)=lna+lnb  好处2.自然对数是严格单调递增函数，对似然函数不会产生任何损失。
# 好处2的根本原因是：因为：自然对数是严格单调递增函数，所以：f(x) 与 ln(f(x)) 的单调性完全相同！
# p1Vect = log(p1Num / p1Denom)   # 矩阵除以一个浮点数，利用Numpy向量很容易实现，列表却难以实现。
# p0Vect = log(p0Num / p0Denom)

# 4-3 朴素贝叶斯分类函数


def classifyNB0(vec2Classify, p0Vec, p1Vec, pClass1):   # 多分类：类别概率用向量表示
    p1 = sum((vec2Classify * p1Vec)) + log(pClass1)
    p0 = sum((vec2Classify * p0Vec)) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def classifyNB1(vec2Classify, pVect, PCategory, slCategory):   # 多分类：类别概率用向量表示
    m = len(slCategory)
    p_vec2Classify = zeros(m)  # ndarray
    for j in range(m):
        p_vec2Classify[j] = sum((vec2Classify * pVect[j])) + log(PCategory[j])
    # 返回概率最大的对应类别
    return slCategory[p_vec2Classify.argsort()[-1]]    # argsort(ndarray)


def testingNB():
    # 主要就是把数据准备、计算相关概率、判断分类等函数综合到一起来。
    # 1 载入数据集
    listOPosts, listClasses = loadDataSet()
    # 2 建立词汇表
    myVocabList = createVocaList(listOPosts)
    # 3 将文本转换成矩阵
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 4 计算条件概率、先验概率
    p0V, p1V, pAv = trainNB0(trainMat, listClasses)
    pVect, PCategory, slCategory = trainNB1(trainMat, listClasses)
    # 5 测试待分类文本
    testEntry = ['love', 'my', 'dalmation']
    thsDoc = array(setOfWords2Vec(myVocabList, testEntry))   # 矩阵便利，批量特征值计算
    print(testEntry, 'trainNB0 classfied as ', classifyNB0(thsDoc,  p0V, p1V, pAv))
    print(testEntry, 'trainNB1 classfied as ', classifyNB1(thsDoc, pVect, PCategory, slCategory))
    testEntry = ['stupid', 'garbage']
    thsDoc = array(setOfWords2Vec(myVocabList, testEntry))  # 矩阵便利，批量特征值计算
    print(testEntry, 'trainNB0 classfied as ', classifyNB0(thsDoc, p0V, p1V, pAv))
    print(testEntry, 'trainNB1 classfied as ', classifyNB1(thsDoc, pVect, PCategory, slCategory))

# 4.5.4 准备数据：文档词袋模型
# 在词集模型中，每个文档中词只能出现一次；而在词袋模型中，每个文档中词可以出现多次。
# 客户购物记录是这样的：['A','S','D','A','S','D','D','F','S','A','D','S'],重复，所以维度大。
# 4-4 朴素贝叶斯词袋模型


def bagofWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1   # returnVec 和 vocabList 词序一致。
    return returnVec


# 每个类别对应的特征词不一样，而且特征权重也不一样，每个文档数目也不一样，高频词去除了吗？
# 以上问题该怎么解决呢！！！能不能为每一个类单独用PCA或者NB？

# 4.6 示例：使用朴素贝叶斯过滤垃圾邮件。
# 4.6.1 准备数据： 切分文本 tsring.split()会把标点符号当成词的一部分。所以，最好用正则表达式。
# 分隔符为除单词、数字外的任意字符串。但容易产生空字符串，可以计算每个字符串的长度，只返回长度>0的字符串。
# 字母大小写转换：1 查找句子的无必要  2 查找单词的话，需转换。(.lower()) 或 (.upper())
# 4.6.2 测试算法： 使用朴素贝叶斯进行交叉验证（集成了：文本解析器）
# 4-5 文件解析及完整的垃圾邮件测试函数


def textParse(bigString):  # 可以添加更多文件解析操作，自己修改。
    import re
    listOfTokens = re.split(r'\w*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    # fullText = []
    for i in range(1, 26):
        # 1 导入并解析文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)  # 一个[]表示一封邮件
        # fullText.extend(wordList)  # 所有邮件的词汇构成词汇表
        classList.append(1)  # 每封待测邮件一个类别标号
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)  # 一个[]表示一封邮件
        # fullText.extend(wordList)  # 所有邮件的词汇构成词汇表
        classList.append(0)  # 每封待测邮件一个类别标号
    # 2 创建词汇表
    vocabList = createVocaList(docList)
    trainingSet = range(50)
    testSet = []
    # 随机构建训练集,从其中随机选择10个作为测试集，实行交叉验证。
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 先选出10个测试邮件序号，然后在doclist中取文件。2个循环，这样好嘛...
        # 先选出测试10个，剩下的就是训练集了。要先训练后测试。如果是2层嵌套循环，不便于单独计时算法训练时间。
        testSet.append(trainingSet[randIndex])
        #  del(trainingSet[randIndex])'range' object doesn't support item deletion
        delete(trainingSet, randIndex, 0)
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagofWords2VecMN(vocabList, docList[docIndex]))  # list index out of range
        trainClasses.append(classList[docIndex])
    p0V, p1V, pAv = trainNB0(array(trainMat), array(trainClasses))
    pVect, PCategory, slCategory = trainNB1(array(trainMat), array(trainClasses))
    errorCount0 = 0
    errorCount1 = 0
    for docIndex in testSet:
        wordVector = bagofWords2VecMN(vocabList, docList[docIndex])
        if classifyNB0(array(wordVector), p0V, p1V, pAv) != classList[docIndex]:
            errorCount0 += 1
        if classifyNB1(array(wordVector), pVect, PCategory, slCategory) != classList[docIndex]:
            errorCount1 += 1
    print(" the classifyNB0() error rate is: ", float(errorCount0) / len(testSet))
    print(" the classifyNB1() error rate is: ", float(errorCount1) / len(testSet))


# 4.7 示例;使用朴素贝叶斯分类器从个人广告中获取区域倾向
# 4.7.1 收集数据：导入RSS数据源
# RSS阅读器：http://code.google.com/p/feedparser/下载安装包，然后解压，切换到解压文件的位置，>>python setup.py install
# anoconda中python也要能够使用才行，暂时不安装。
















