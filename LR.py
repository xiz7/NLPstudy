# coding: utf-8
# linear_regression/regression.py
import numpy as np
import time

class LinearRegression(object):
    def exeTime(func):
        """ 耗时计算装饰器
        """
        def newFunc(*args, **args2):
            t0 = time.time()
            back = func(*args, **args2)
            return back, time.time() - t0
        return newFunc
    
    def loadDataSet(self,filename):
        """ 读取数据
    
        从文件中获取数据，在《机器学习实战中》，数据格式如下
        "feature1 TAB feature2 TAB feature3 TAB label"
    
        Args:
            filename: 文件名
    
        Returns:
            X: 训练样本集矩阵
            y: 标签集矩阵
        """
        numFeat = len(open(filename).readline().split(',')) - 1
        
        X = []
        y = []
        file = open(filename)
        for line in file.readlines():
           
            lineArr = []
            curLine = line.strip().split(',')
            
            for i in range(numFeat):
                
                lineArr.append(float(curLine[i]))
            X.append(lineArr)
            y.append(float(curLine[-1]))
        return np.mat(X), np.mat(y).T
    
    def h(self,theta, x):
        """预测函数
    
        Args:
            theta: 相关系数矩阵
            x: 特征向量
    
        Returns:
            预测结果
        """
        return (theta.T*x)[0,0]
    
    def J(self,theta, X, y):
        """代价函数
    
        Args:
            theta: 相关系数矩阵
            X: 样本集矩阵
            y: 标签集矩阵
    
        Returns:
            预测误差（代价）
        """
        m = len(X)
        return (X*theta-y).T*(X*theta-y)/(2*m)
    
    @exeTime
    def bgd(self,rate, maxLoop, epsilon, X, y):
        """批量梯度下降法
    
        Args:
            rate: 学习率
            maxLoop: 最大迭代次数
            epsilon: 收敛精度
            X: 样本矩阵
            y: 标签矩阵
    
        Returns:
            (theta, errors, thetas), timeConsumed
        """
        m,n = X.shape
        # 初始化theta
        theta = np.zeros((n,1))
        count = 0
        converged = False
        error = float('inf')
        errors = []
        thetas = {}
        for j in range(n):
            thetas[j] = [theta[j,0]]
        while count<=maxLoop:
            if(converged):
                break
            count = count + 1
            for j in range(n):
                deriv = (y-X*theta).T*X[:, j]/m
                theta[j,0] = theta[j,0]+rate*deriv
                thetas[j].append(theta[j,0])
            error = self.J(theta, X, y)
            errors.append(error[0,0])
            # 如果已经收敛
            if(error < epsilon):
                converged = True
        return theta,errors,thetas
    
    
    @exeTime
    def sgd(self,rate, maxLoop, epsilon, X, y):
        """随机梯度下降法
        Args:
            rate: 学习率
            maxLoop: 最大迭代次数
            epsilon: 收敛精度
            X: 样本矩阵
            y: 标签矩阵
        Returns:
            (theta, error, thetas), timeConsumed
        """
        m,n = X.shape
        # 初始化theta
        theta = np.zeros((n,1))
        count = 0
        converged = False
        error = float('inf')
        errors = []
        thetas = {}
        for j in range(n):
            thetas[j] = [theta[j,0]]
        while count <= maxLoop:
            if(converged):
                break
            count = count + 1
            errors.append(float('inf'))
            for i in range(m):
                if(converged):
                    break
                diff = y[i,0]-self.h(theta, X[i].T)
                for j in range(n):
                    theta[j,0] = theta[j,0] + rate*diff*X[i, j]
                    thetas[j].append(theta[j,0])
                error = self.J(theta, X, y)
                errors[-1] = error[0,0]
                # 如果已经收敛
                if(error < epsilon):
                    converged = True
        return theta, errors, thetas
    
    
    def standardize(X):
        """特征标准化处理
    
        Args:
            X: 样本集
        Returns:
            标准后的样本集
        """
        m, n = X.shape
        # 归一化每一个特征
        for j in range(n):
            features = X[:,j]
            meanVal = features.mean(axis=0)
            std = features.std(axis=0)
            if std != 0:
                X[:, j] = (features-meanVal)/std
            else:
                X[:, j] = 0
        return X

    def normalize(X):
        """特征归一化处理
    
        Args:
            X: 样本集
        Returns:
            归一化后的样本集
        """
        m, n = X.shape
        # 归一化每一个特征
        for j in range(n):
            features = X[:,j]
            minVal = features.min(axis=0)
            maxVal = features.max(axis=0)
            diff = maxVal - minVal
            if diff != 0:
               X[:,j] = (features-minVal)/diff
            else:
               X[:,j] = 0
        return X
# =============================================================================
# if __name__ == "__main__": 
#     LR= LinearRegression()
#     print(LR.loadDataSet('data.txt'))
# =============================================================================
