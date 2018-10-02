# -*- coding:utf-8 -*-

import numpy as np


class GBDT():
    """
    cart回归树作弱学习器，平方误差函数作损失函数。Loss = 1/2*(y-h(x))**2
    """

    def caclSE(self, dataSet):
        '''
        计算CART回归树的节点方差Squared Error
        :param dataSet: 数据集，包含目标列。  np.array，m*(n+1)
        :return: 当前节点（目标列）的方差
        '''
        if dataSet.shape[0] == 0:  # 如果输入一个空数据集，则返回0
            return 0
        return np.var(dataSet[:, -1]) * dataSet.shape[0]  # 方差=均方差*样本数量

    def splitDataSet(self, dataSet, feature, value):
        """
        根据给定特征值，二分数据集。
        :param dataSet: 同上
        :param feature: 待划分特征。因为是处理回归问题，这里我们假定数据集的特征都是连续型
        :param value: 阀值
        :return: 特征值小于等于阀值或大于阀值的两个子数据集. k*(n+1), (m-k)*(n+1)
        """
        arr1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]  # 利用np.nonzero返回目标样本的索引值
        arr2 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
        return arr1, arr2

    def chooseBestFeature(self, dataSet):
        """
        通过比较所有节点的方差和，选出方差和最小的特征与对应的特征值
        :param dataSet: 同上
        :return: 最佳划分点和划分值
        """
        n = dataSet.shape[1] - 1  # m是样本数量，n是特征数量
        minErr = np.inf  # 初始化最小方差为无穷大的正数
        bestFeature, bestValue = 0, 0  # 声明变量的类型
        for feature in range(n):
            values = set(dataSet[:, feature].tolist())  # 选取所有出现过的值作为阀值
            for value in values:
                arr1, arr2 = self.splitDataSet(dataSet, feature, value)
                err1 = self.caclSE(arr1)
                err2 = self.caclSE(arr2)
                newErr = err1 + err2
                # 选取方差和最小的特征和对应的阀值
                if newErr < minErr:
                    minErr = newErr
                    bestFeature = feature
                    bestValue = value
        return bestFeature, bestValue

    def calcLeaf(self, dataSet):
        """
        计算当前节点的目标列均值（作为当前节点的预测值）
        预测值的计算具体是要根据损失函数确定的。
        不用的损失函数，对应不同的叶子节点值。
        平方误差损失的节点值是均值。
        :param dataSet: 同上
        :return: 目标列均值
        """
        return np.mean(dataSet[:, -1])

    def createTree(self, dataSet, max_depth=4):
        """
        创建CART回归树
        :param dataSet: 同上
        :param max_depth: 设定回归树的最大深度，防止无限生长（过拟合）
        :return: 字典形式的cart回归树模型
        """
        if len(set(dataSet[:, -1].tolist())) == 1:  # 如果当前节点的值都相同，结束递归
            return self.calcLeaf(dataSet)
        if max_depth == 1:  # 如果层数超出设定层数，结束递归
            return self.calcLeaf(dataSet)
        # 创建回归树
        bestFeature, bestValue = self.chooseBestFeature(dataSet)
        mytree = {}
        mytree['FeatureIndex'] = bestFeature  # 存储分割特征值的索引
        mytree['FeatureValue'] = bestValue  # 存储阀值
        lSet, rSet = self.splitDataSet(dataSet, bestFeature, bestValue)
        mytree['left'] = self.createTree(lSet, max_depth - 1)  # 存储左子树的信息
        mytree['right'] = self.createTree(rSet, max_depth - 1)  # 存储右子树的信息

        return mytree

    def predict_byCart(self, cartTree, testData):
        """
        根据训练好的cart回归树，预测待测数据的值
        :param cartTree: 训练好的cart回归树
        :param testData: 待测试数据, 1*n
        :return: 预测值
        """
        if not isinstance(cartTree, dict):  # 不是字典，意味着到了叶子结点，此时返回叶子结点的值即可
            return cartTree
        featureIndex = cartTree['FeatureIndex']  # 获取回归树的第一层特征索引
        featureVal = testData[featureIndex]  # 根据特征索引找到待测数据对应的特征值， 作为下面是进入左子树还是右子树的依据
        if featureVal <= cartTree['FeatureValue']:
            return self.predict_byCart(cartTree['left'], testData)
        elif featureVal > cartTree['FeatureValue']:
            return self.predict_byCart(cartTree['right'], testData)

    def predict_all_byCart(self, cartTree, testData):
        """
        根据训练好的cart回归树预测所有待测数据的值
        :param cartTree: 同上
        :param testData: 待测试数据，m*n
        :return: 预测值，1*m
        """
        testData = np.array(testData)
        predictions = np.zeros(testData.shape[0])
        for i in range(testData.shape[0]):
            predictions[i] = self.predict_byCart(cartTree, testData[i])
        return predictions

    def GBDTtraining(self, dataSet, numIt=4):
        """
        训练GBDT回归树,简化版本，这里没根据Shrinkage思想添加正则化参数step
        :param dataSet: 同上
        :param numIt: 梯度下降轮次，即生成弱学习器的个数
        :return: cart回归树组成的列表
        """
        fx_pre = np.mean(dataSet[:, -1])  # 记录前m-1个模型的预测值之和
        weakRegArr = []  # 弱学习器的列表
        targets = dataSet[:, -1].copy()  # 存储训练集的目标值
        for i in range(numIt):
            dataSet[:, -1] = targets - fx_pre  # 平方误差损失的一阶导的负数（负梯度）正好是目标值与预测值的差
            mytree = self.createTree(dataSet)  # 把原目标列替换为残差后，训练弱学习器
            weakRegArr.append(mytree)
            fx_pre += self.predict_all_byCart(mytree, dataSet[:, :-1])  # 计算所有模型的预测值之和（针对训练集）
            loss = np.var(targets - fx_pre) * targets.shape[0]  # 计算损失函数
            print('Iter:%d, Loss: %.6f' % (i + 1, loss))
            if loss == 0:  # 损失为0，跳出循环。或者可以设定一个阀值，当小于该阀值的时候，退出循环
                break
        return weakRegArr

    def predict(self, GBDTtree, testData):
        """
        根据训练好的GDBT模型，预测待测数据
        :param GBDTtree: 训练好的GBDT模型
        :param testData: 待测数据m*n, m>=1
        :return: 预测值
        """
        testData = np.array(testData)
        predict = GBDTtree[0]  # 模型的初始化值f0x
        for cart in GBDTtree[1:]:  # cart树从索引1开始
            predict += self.predict_all_byCart(cart, testData)  # 累加预测结果
        return predict


# 以下是测试数据
dataSet = np.array([[1, 5.56],
                    [2, 5.70],
                    [3, 5.91],
                    [4, 6.40],
                    [5, 6.80],
                    [6, 7.05],
                    [7, 8.90],
                    [8, 8.70],
                    [9, 9.00],
                    [10, 9.05]])
test = dataSet[:, 0].reshape(10, 1)

gbdtTree = GBDT()
# mytree = gbdtTree.createTree(dataSet)
# print(mytree)
gbdt = gbdtTree.GBDTtraining(dataSet)
print(gbdtTree.predict(gbdt, [[2], [5]]))
