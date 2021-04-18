'''
Author: xiaomingaaa
Date: 2021-04-16 19:37:02
LastEditTime: 2021-04-17 20:36:09
LastEditors: Please set LastEditors
Description: 1. 使用sklearn中自带的iris数据集做分类任务 2. 扩展adaboost方法在生成的数据上做回归任务
FilePath: /ml/ensemble learning/adaboost.py
'''
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=10, num_hidden=3, activation='Relu'):
        super.__init__(MLP, self)
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.activation = activation
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.ModuleList()
        self.MLP_Net()

    def MLP_Net(self):
        if self.activation == 'Relu':
            activation = nn.ReLU()
        else:
            activation = nn.LeakyReLU()
        dims = [self.in_dim]
        for i in range(self.num_layers):
            dims.append(self.hidden_dim)
        dims.append(self.out_dim)
        for i in range(len(dims)-1):
            self.mlp.append(nn.Linear(dims[i], dims[i+1]))
            self.mlp.append(activation)
        self.mlp.pop()

    def forward(self, X, y):
        for layer in self.mlp:
            X = layer(X)
        loss = F.cross_entropy(X, y)
        return loss, F.sigmoid(X)


def load_data(e_num):
    iris = load_iris()
    if e_num > len(iris.data):
        e_num = len(iris.data)
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width',
                  'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:e_num, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    return data[:, :2], data[:, -1]


def split_data(X, y, shuffle_seed=123, test_size=0.2):
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=test_size, random_state=123)
    return train_X, train_y, test_X, test_y


class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        # 变换分类阈值的步数
        self.learning_rate = learning_rate

    '''
    @description: 初始化模型参数
    @param {*} self
    @param {*} X 训练样本
    @param {*} y 标签
    @return {*}
    '''

    def init(self, X, y):
        self.X = X
        self.y = y
        self.M, self.N = X.shape
        self.cls_list = []
        # 初始均匀分布
        self.weights = [1.0/self.M]*self.M
        # 初始分类器G(x)系数
        self.alphas = []
    '''
    @description: 实现决策树为当前的基分类器
    @param {*} self
    @param {*} features
    @param {*} labels
    @param {*} weights
    @return {*}
    '''

    def decision_tree(self, features, labels, weights):
        pass

    '''
    @description: base分类器，这里根据李航统计学习AdaBoost样例实现, 使用了一个决策树桩（一层）
    @param {*} self
    @param {*} features 样本的单维特征，根据这个单维特征选择最优分类特征
    @param {*} labels 样本标签
    @param {*} weights 样本权重
    @return {*} best_v, direct, error (误差期望), compare_array
    '''

    def General_G(self, features, labels, weights):
        m = len(features)
        error = sys.maxsize
        best_v = 0.0
        # M x 1
        feature_min = min(features)
        feature_max = max(features)
        max_step = (feature_max-feature_min +
                    self.learning_rate)//self.learning_rate
        direct, compare_array = None, None
        for i in range(1, int(max_step)):
            # current threshold
            v = feature_min+self.learning_rate*i
            if v not in features:
                compare_array_positive = np.array(
                    [1 if features[k] > v else -1 for k in range(m)])
                # 统计分类错误的样本weights
                weights_error_positive = sum([weights[k] for k in range(
                    m) if compare_array_positive[k] != labels[k]])

                compare_array_negative = np.array(
                    [-1 if features[k] > v else 1 for k in range(m)])
                weights_error_negative = sum([weights[k] for k in range(
                    m) if compare_array_negative[k] != labels[k]])
                if weights_error_positive < weights_error_negative:
                    weight_error = weights_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weights_error_negative
                    _compare_array = compare_array_negative
                    direct = 'negative'
                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v

        return best_v, direct, error, compare_array

    '''
    @description: 计算分类起的alpha系数
    @param {*} self
    @param {*} error 错误率
    @return {*}
    '''

    def calc_alpha(self, error):
        return np.log((1-error)/error)*0.5

    '''
    @description: 计算规范化因子 
    @param {*} self
    @param {*} weights
    @param {*} a
    @param {*} clf_result
    @return {*}
    '''

    def calc_z(self, weights, a, clf_result):
        return sum([weights[i] * np.exp(-1*a * self.y[i] * clf_result[i]) for i in range(self.M)])

    '''
    @description: 更新样本权重
    @param {*} self
    @param {*} z
    @param {*} a
    @param {*} clf_result
    @return {*}
    '''

    def update_w(self, z, a, clf_result):
        for i in range(self.M):
            self.weights[i] = self.weights[i] * \
                np.exp(-a*self.y[i]*clf_result[i])

    def fit(self, X, y):
        self.init(X, y)
        for epoch in range(self.n_estimators):
            best_clf_error, best_v, clf_result = sys.maxsize, None, None
            for j in range(self.N):
                v, direct, error, compare_array = self.General_G(
                    self.X[:, j], self.y, self.weights)
                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    # 记录分类最优的特征
                    axis = j
                if best_clf_error == 0:
                    # 全部分类正确，不需要在搜索其他类型特征
                    break
        a = self.calc_alpha(best_clf_error)
        self.alphas.append(a)
        # record args (j-th column) of current classifiler
        self.cls_list.append((axis, best_v, direct))
        z = self.calc_z(self.weights, a, clf_result)
        self.update_w(z, a, clf_result)

    def forward_G(self, v, feature, direct):
        if direct == 'positive':
            return 1 if feature > v else -1
        else:
            return -1 if feature > v else 1

    def predict(self, feature):
        result = 0.0
        for i in range(len(self.cls_list)):
            axis, v, direct = self.cls_list[i]
            f_input = feature[axis]
            result += self.alphas[i]*self.forward_G(v, f_input, direct)
        return 1 if result > 0 else -1

    def score(self, X_test, y_test):
        acc_count = 0
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                acc_count += 1
        return acc_count/len(X_test)


def main():
    example_size = 100
    X, y = load_data(example_size)
    train_X, train_y, test_X, test_y = split_data(X, y)
    adaboost = AdaBoost(n_estimators=100, learning_rate=0.1)
    adaboost.fit(train_X, train_y)
    print(adaboost.score(test_X, test_y))


if __name__ == '__main__':
    main()
