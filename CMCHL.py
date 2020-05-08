import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import svd
from sklearn.preprocessing import OneHotEncoder
import argparse
import pandas as pd
from sklearn import preprocessing
import random

class HEMAP:
    def __init__(self, src_data, tar_data, args):
        # src_data and tar_data types: numpy
        self.src_value = src_data.iloc[:, :-2].values
        self.tar_value = tar_data.iloc[:, :-2].values
        self.src_label = src_data.iloc[:, -2:].values
        self.tar_label = tar_data.iloc[:, -2:].values
        self.src_index = src_data.index.tolist()
        self.tar_index = tar_data.index.tolist()
        self.beta = args[0]
        self.theta = 1 - args[0]
        self.topk = args[1]
        self.eExpected = args[2]
        self.steps = args[3]
        self.learningRate = args[4]

        self.src_predcate, self.tar_predcate = self.generate_partition_matrix()
        self.src_feature = np.concatenate((self.src_value, self.src_predcate), axis=1)
        self.tar_feature = np.concatenate((self.tar_value, self.tar_predcate), axis=1)
        self.src_grad, self.tar_grad = self.gradientDescend()
        self.src_projected, self.tar_projected = self.make_projected_data()

    @staticmethod
    def one_hot(x):
        x = x.reshape([-1, 1])
        encoder = OneHotEncoder(categories='auto')
        encoder.fit(x)
        onehot_labels = encoder.transform(x).toarray()
        return onehot_labels

    def generate_partition_matrix(self):
        src_neigh = KMeans(n_clusters=2)
        src_neigh.fit(self.src_value)
        src_partition = src_neigh.predict(self.src_value)
        src_partition = src_partition.reshape([-1, 1])
        scaler_source = preprocessing.StandardScaler().fit(src_partition)
        source_label = scaler_source.transform(src_partition)

        tar_neigh = KMeans(n_clusters=2)
        tar_neigh.fit(self.tar_value)
        tar_partition = tar_neigh.predict(self.tar_value)
        tar_partition = tar_partition.reshape([-1, 1])
        scaler_target = preprocessing.StandardScaler().fit(tar_partition)
        target_label = scaler_target.transform(tar_partition)

        src_partition_col = source_label.reshape([-1, 1])
        tar_partition_col = target_label.reshape([-1, 1])

        return src_partition_col, tar_partition_col

    def gradientDescend(self):
        mT, nT = self.tar_feature.shape[0], self.tar_feature.shape[1]
        mS, nS = self.src_feature.shape[0], self.src_feature.shape[1]
        bT, pT = np.random.rand(mT, self.topk), np.random.rand(self.topk, nT)
        bS, pS = np.random.rand(mS, self.topk), np.random.rand(self.topk, nS)
        step = 0
        while step < self.steps:
            multiArray=np.matmul
            ppT = multiArray(pT, pT.T)
            ppS = multiArray(pS, pS.T)
            print(multiArray(bT,pT).shape,self.tar_feature.shape,bT.shape,pT.shape,ppT.shape,self.learningRate)
            bT = bT - 2 * (self.learningRate) * (multiArray(bT,ppT) - multiArray(self.tar_feature,pT.T) + self.beta*(bT - bS))
            bS = bS - 2 * (self.learningRate) * (multiArray(bS,ppS) - multiArray(self.src_feature,pS.T) + self.beta*(bS - bT))
            pT = (self.learningRate) * multiArray(multiArray(np.power(multiArray(bT.T,bT), -1),bT.T),self.tar_feature)
            pS = (self.learningRate) * multiArray(multiArray(np.power(multiArray(bS.T,bS), -1),bS.T),self.src_feature)
            Fnorm = np.linalg.norm
            e = self.theta*Fnorm(multiArray(bT,pT) - self.tar_feature, ord=2) + self.theta*Fnorm(multiArray(bS,pS) - self.src_feature, ord=2) + self.beta * Fnorm(bS-bT, ord=2)
            print(step,e)
            if e <= self.eExpected:
                break
            step += 1
        return bS, bT

    def make_projected_data(self):
        values = np.concatenate((self.src_grad, self.src_label), axis=1)
        valuet = np.concatenate((self.tar_grad, self.tar_label), axis=1)

        src_projected = pd.DataFrame(values, index=self.src_index)
        tar_projected = pd.DataFrame(valuet, index=self.tar_index)

        return src_projected, tar_projected


if __name__ == '__main__':
    tar = pd.read_csv("./regeneration/sorted/targetRegenerationPCA.csv", index_col=0)
    src = pd.read_csv("./regeneration/sorted/sourceRegenerationPCA.csv", index_col=0)
    # tarNum,srcNum=tar.index.tolist(),src.index.tolist()
    # random.shuffle(srcNum)
    # random.shuffle(tarNum)
    # tarRandom,srcRandom=tar.iloc[tarNum],src.iloc[srcNum]
    arg = [0.4,6,1,500,0.1]
    print(arg)
    p1 = HEMAP(src, tar, arg)
    print(p1.src_projected)
    print(p1.tar_projected)
    p1.src_projected.to_csv("./projectedData/grad/b0.4/6/srcProjected.csv", index=True)
    p1.tar_projected.to_csv("./projectedData/grad/b0.4/6/tarProjected.csv", index=True)
