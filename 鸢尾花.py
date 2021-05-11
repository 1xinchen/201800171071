import random
from numpy import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据集
# 鸢尾花(iris)数据集
# 数据集内包含 3 类共 150 条记录，每类各 50 个数据，
# 每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
# 可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。
# 这里只取前100条记录，两项特征，两个类别。
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]

# 用于在某个区间范围内随机选择一个整数
def selectJrand(i, m):  # i为第一个alpha下标，m是所有alpha对应的数目
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

# 用于调整大于H或小于L的alpha值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 测试
dataArr, labelArr = create_data()
#print(dataArr)

# 简化版SMO算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  # 数据集、类别标签、常数C、容错率和退出前最大的循环次数
    dataMatrix = mat(dataMatIn)  # 将数据集用mat函数转换为矩阵
    labelMat = mat(classLabels).transpose()  # 转置类别标签（使得类别标签向量的每行元素都和数据矩阵中的每一行一一对应）
    b = 0
    m, n = shape(dataMatrix)  # 通过矩阵的shape属性得到常数m,n
    alphas = mat(zeros((m, 1)))  # 构建一个alpha列矩阵，初始化为0
    iter = 0  # 建立一个iter变量（该变量存储在没有任何alpha改变的情况下遍历数据集的次数，当该变量达到输入值maxIter时，函数结束）
    while (iter < maxIter):  # 遍历数据集的次数小于输入的maxIter
        alphaPairsChanged = 0  # 用于记录alpha是否已经进行优化，先设为0
        for i in range(m):  # 顺序遍历整个集合
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # fXi可以计算出来，是我们预测的类别
            Ei = fXi - float(labelMat[i])  # 计算误差，基于这个实例的预测结果和真实结果的比对
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or (
                    (labelMat[i] * Ei > toler) and (alphas[i] > 0)):  # 如果误差过大，那么对该数据实例对应的alpha值进行优化
                j = selectJrand(i, m)  # 随机选取第二个alpha，调用辅助函数selectJrand
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])  # 计算误差（同第一个alpha值的误差计算方法）
                alphaIold = alphas[i].copy()  # 浅复制，两对象互不影响，为了稍后对新旧alpha值进行比较
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):  # 保证alpha值在0与C之间
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    #print("L==H")
                    continue  # 本次循环结束，进入下一次for循环
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T  # alpha值的最优修改值
                if eta >= 0:
                    #print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta  # 计算出新的alpha[j]值
                alphas[j] = clipAlpha(alphas[j], H, L)  # 调用clipAlpha辅助函数以及L,H值对其进行调整
                if (abs(alphas[j] - alphaJold) < 0.00001):  # 检查alpha[j]是否有轻微改变，如果是的话，就退出for循环
                    #print("J not moving enough!")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # 对alpha[i]进行同样的改变，改变大小一样方向相反

                # 给alpha[i]和alpha[j]设置一个常数项b，等同于上篇博客中计算阀值b部分，可以对照看
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1  # 当for循环运行到这一行，说明已经成功的改变了一对alpha值，同时让alphaPairsChanged加1
                #print("iter:%d i:%d,pairs changed %d" % (iter, i, alphaPairsChanged))
        # for循环结束后，检查alpha值是否做了更新
        if (alphaPairsChanged == 0):  # 如果没有，iter加1
            iter += 1  # 下面后回到while判断
        else:  # 如果有更新则iter设为0后继续运行程序
            iter = 0
        #print("iteration number: %d" % iter)
    return b, alphas  # （只有在所有数据集上遍历maxIter次，且不再发生任何alpha值修改之后，程序才会停止，并退出while循环）

# 测试
b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print('b的取值为：',float(b[0,0]))
#print(alphas[alphas > 0])  # 观察元素大于0的

shape(alphas[alphas > 0])  # 得到支持向量的个数
"""
for i in range(100):
    if alphas[i] > 0.0:
        print("简单版支持向量为:", dataArr[i], labelArr[i])
"""

# 计算w的值
# 该程序最重要的是for循环，for循环中实现的仅仅是多个数的乘积，前面我们计算出的alpha值，大部分是为0
# 虽然遍历数据集中的所有数据，但是起作用的只有支持向量，其他对计算w毫无作用
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


ws = calcWs(alphas, dataArr, labelArr)
print('w0的取值：',float(ws[0]))
print('w1的取值：',float(ws[1]))



#主程序
xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers = []
colors = []
for i in range(0,99,1):
    xPt=float(dataArr[i,0])
    yPt=float(dataArr[i,1])
    label=int(labelArr[i])
    if (label == -1):
        xcord0.append(xPt)
        ycord0.append(yPt)
    else:
        xcord1.append(xPt)
        ycord1.append(yPt)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0, ycord0, marker='s', s=90)
ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
plt.title('Support Vectors Circled')
"""# plt.plot([2.3,8.5], [-6,6]) #seperating hyperplane
b = -4.996000000000346
w0 = 2.2200000000000832
w1 = -2.2200000000000353
x = arange(3, 8, 0.01)
y = (-w0 * x - b) / w1
ax.plot(x, y)
ax.axis([3, 8, 1, 5])"""
plt.show()



