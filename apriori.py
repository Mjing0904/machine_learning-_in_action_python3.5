# 11-1 Apriori算法中的辅助函数


# 每条交易记录商品数目是不等的
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 怎么找到支持度＞0.8的所有项集？
# 如果建立一个所有商品的可能组合的清单，然后对每一组合统计出它的频繁程度，那么有上千万的商品时，这种方法非常慢！
# 怎么做才好呢？apriori原理！
# 但是，购物记录怎么表示好？就好像词向量模型怎么建立比较好？
# 如果每条购物记录是维度为所有商品的总数，每种商品的元素值是该商品的购买数目，那么用矩阵批处理很好，
# 但是，这样很耗费存储空间啊！对比NB朴素贝叶斯的改进模型？但是，怎么统计词频率？

# Apriori原理：向下封闭性：
# 如果某个项集是频繁的，那么它的所有子集也是频繁的。
# 反之，如果某个项集是非频繁的，那么它的所有超集都是非频繁的。
# 这样，所有非频繁项集的超集的支持度都不需要计算了，可以避免项集数目的指数增长，大大减少了计算的数量。

# 建立一个所有商品的不可变集合

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            # 2个不重复：频繁项集不重复、项集内元素不重复，所以，结构是：集合的集合！
            # 但是，python中集合不能嵌套！频繁项集的操作根本是集合的运算。
            # 为了方便以后的频繁项集的合并，所以初始的集合元素必须是{单商品}！
            # 列表元素可以是集合！如：s=[{1,2,3},{2,3,4}],s[0]|s[1]集合的或操作，len(s)==2求长度方便！
            # 列表元素可以重复！如：s=[{1,2,3},{2,3,4},{2,3,4}]，所以，结构得是{{}}
            # python可以创建只有一个整数的集合。如:q={1}
            # frozenset(p) p必须是可迭代对象，如[]、{}. 如：o=frozenset([2]) o1=frozenset({2}) o==o1 --> True
            # o==o1,所以frozenset({item})==frozenset([item]),都可以使用！！！
            if frozenset([item]) not in C1:  # if not [item] in C1: 从源头开始查找问题，换思路！
                # s=[{1,3}] s.append({2,4})-> s=[{1,3},{2,4}] s.extend({2,4})-->s=[{1,3},2,4] 不排序！
                C1.append(frozenset([item]))
                # [frozenset({1}), frozenset({3}), frozenset({4}), frozenset({2}), frozenset({5})]
                # if not [item] in C1:
                #     C1.append([item])
    C1.sort()  # set会自动从小到大排序的，列表不会。
    # p=[1,1,1,2,3,2] p1=frozenset(p) p1: frozenset({1, 2, 3})  p2=set(p) p2: {1, 2, 3}
    return C1  # 因为集合C1要做字典的键值，只有frozenset可以实现,map(frozenset, C1)对每个C1元素执行frozenset()。


# 筛选出满足最小支持度的项集
def scanD(D, Ck, minSupport):
    i = 0
    ssCnt = {}
    # 对于数据集中的每条记录，候选集中的每个候选项，利用字典数据结构统计其频度！
    for tid in D:
        # print("我是tid: ", tid)  # print("我是tid: " + tid) :Can't convert 'list' object to str implicitly
        for can in Ck:
            # i += 1 加入该语句发现问题出现在内循环。
            # i += 1
            # print("我是can: ", can)
            # print("你为什么不循环？", i)
            # can.issubset(tid) can、tid都得是集合！ set([1,3,4])=={1,3,4} frozenset([1,3,4])==frozenset({1,3,4})
            if can.issubset(set(tid)):
                # print("can issubset set(tid)？: ", can.issubset(set(tid)))
                if not ssCnt.__contains__(can):
                    # print("ssCnt contains can？: ", ssCnt.__contains__(can))
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))  # D=map(set,dataSet) object of type 'map' has no len()
    # 计算所有项集的支持度
    retList = []
    supportData = {}
    # print(ssCnt)
    # 为什么不执行！ssCnt空！
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)  # retList=[]  retList.insert(0,'A')-> ['A'] retList.insert(0,'B')->['B', 'A']
        supportData[key] = support
    return retList, supportData


# 筛选出满足最小支持度的项集
# D=[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
# Ck=map(frozenset, [{1}, {3}, {4}, {2}, {5}])
#     for tid in D:
#         for can in Ck:          #内循环一遍，外循环4遍！
#
#     for tid in D:               #内循环4遍，外循环4遍！
#         for can in map(frozenset, [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]):
#
#     for tid in [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]:  #内循环4遍，外循环4遍！
#         for can in map(frozenset, [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]):
#
#     for tid in [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]:   #内循环4遍，外循环4遍！
#         for can in [frozenset({1}), frozenset({3}), frozenset({4}), frozenset({2}), frozenset({5})]:
#  总结：1.Ck=map(frozenset, [{1}, {3}, {4}, {2}, {5}]) 是map类型，而map最近最短时间迭代一次后就退出迭代。
# 2. for can in map(frozenset, [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}])
#     此内循环每次都会执行map一次，即每次都会迭代一遍，所以，与for can in Ck 循环结果不同！
# 3.肖潇，有3个重要的思维优势：1.助人为乐，这是一种共赢思维！2.当出现问题时，她会从源头开始重新查找问题，大局观！
# 3.当我给肖潇解释map只迭代一次后，因为之前我误导肖潇map是[]列表形式，依然是可迭代的，所以，肖潇立马质疑：
# 列表是可迭代的，map是列表形式的，那应该可以被迭代多次啊，还是不能解释：为什么map只迭代一次！(细节逻辑，肖潇比我强！)
# 所以，后来我验证map(func,iter)==[func(iter0),...,func(itern)] False! 所以，map是一个特殊类型！后来，我将map(func,iter)
# 功能放到程序代码前一段有的for循环中，终于解决了问题。事实上，return map(func,iter)就已经包括循环了，代码功能重复！
# 4.上善若水网友：同样看class map 的英文说明，但是他注意到最后一句话，注重细节的人，学习！


# 11-2 Apriori算法
# def aprioriGen(Lk, k):
#     retList = []
# k频繁项集的长度，即k频繁项的个数
# lenLk = len(Lk)
# 对k频繁项集每个频繁项两两不重复组合成新的频繁项，是K+1项。
# 怎么直接创建k项集？直接创建，有可能不是由k-1频繁项集产生的k项集，而且产生k项集不一定是k频繁项集！
# Lk[i] | Lk[j] 会产生重复的频繁项集，所以，得转换成集合！之后，为了用index，还得转换成列表！麻烦！
# 所以，要尝试找出直接生成K项集的规律！而且，得减少列表的遍历！！！要写高效的代码！
# for i in range(lenLk):
#     for j in range(i+1, lenLk):
#         tmpset = Lk[i] | Lk[j]
#         # 会生成长度＞k的项集，所以得检查！
#         if tmpset.__len__(tmpset) == k:
#             retList.append(tmpset)
# setretList = set(retList)
# lstretList = list(setretList)
# return lstretList

# 11-2 Apriori算法
def aprioriGen(Lk, k):
    retList = []
    # k频繁项集的长度，即k频繁项的个数
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 1.发现只比较前k-2的元素是否相同，就可以成倍减少列表遍历次数！注意：前k-2的元素也是多个，所以要装进列表！
            # 2.我现在知道为什么用frozenset([2])而不是frozenset({2})了，就是为了[:k - 2]，其实都可以！
            # 我也知道为什么要排序了，就是为了减少遍历次数！
            # eg.[frozenset([1,,2,3])，frozenset([1,2,5])，frozenset([1,3,5])，frozenset([2,3,5])] 遍历次数减少到1/6!
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:  # 列表值相同，引用就是相同的。map不同。
                retList.append(Lk[i] | Lk[j])
    return retList


# 封装操作，用户操作函数
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    # 购物记录要去重，购物集合不用去重。
    # D = map(set, dataSet)
    L1, supportData = scanD(dataSet, C1, minSupport)
    # 妙！ L=[L1， L2， L3，...,LK]
    L = [L1]
    k = 2
    # len(L[k-2])因为k+=1，所以每次L[k-2]都用新的Lk!直至不再产生新的频繁项集，循环结束！
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)  # 初始项集
        Lk, supK = scanD(dataSet, Ck, minSupport)  # 频繁项集
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


# L,supportdate = apriori.apriori(dataSet)
# [[frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})],
# [frozenset({3, 5}), frozenset({1, 3}), frozenset({2, 5}), frozenset({2, 3})],
# [frozenset({2, 3, 5})],
# []]

# 11.4 从频繁项集中挖掘关联规则
# 11-3 关联规则生成函数


def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportdata, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportdata[freqSet] / supportdata[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf: ', conf)
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > (m + 1):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)  # 递归调用，怪不得！
