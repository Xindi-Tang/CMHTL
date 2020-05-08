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
        self.theta = 1-args[0]
        self.topk = args[1]

        self.src_predcate, self.tar_predcate = self.generate_partition_matrix()
        self.src_feature = np.concatenate((self.src_value,self.src_predcate), axis=1)
        self.tar_feature = np.concatenate((self.tar_value, self.tar_predcate), axis=1)
        self.a = self.construct_a_matrix()
        self.U = self.calculate_eigenvalue()
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
        src_partition = src_partition.reshape([-1,1])
        scaler_source = preprocessing.StandardScaler().fit(src_partition)
        source_label = scaler_source.transform(src_partition)

        tar_neigh = KMeans(n_clusters=2)
        tar_neigh.fit(self.tar_value)
        tar_partition = tar_neigh.predict(self.tar_value)
        tar_partition=tar_partition.reshape([-1,1])
        scaler_target = preprocessing.StandardScaler().fit(tar_partition)
        target_label = scaler_target.transform(tar_partition)

        src_partition_col = source_label.reshape([-1,1])
        tar_partition_col = target_label.reshape([-1,1])

        return src_partition_col, tar_partition_col

    def construct_a_matrix(self):
        # c_s, c_t = self.src_predcate, self.tar_predcate
        # a_1 = 2 * self.theta ** 2 * np.matmul(self.tar_feature, np.transpose(self.tar_feature)) + \
        #       self.beta ** 2 / 2 * np.matmul(self.src_feature, np.transpose(self.src_feature)) + \
        #       (1 - self.theta) * (self.beta + 2 * self.theta) * np.matmul(c_t, np.transpose(c_t))
        # a_2 = self.beta * self.theta * (np.matmul(self.tar_feature, np.transpose(self.tar_feature))
        #                                 + np.matmul(self.src_feature, np.transpose(self.src_feature)))
        # a_3 = self.beta * self.theta * (np.matmul(self.tar_feature, np.transpose(self.tar_feature))
        #                                 + np.matmul(self.src_feature, np.transpose(self.src_feature)))
        # a_4 = 2 * self.theta ** 2 * np.matmul(self.src_feature, np.transpose(self.src_feature)) + \
        #       self.beta ** 2 / 2 * np.matmul(self.tar_feature, np.transpose(self.tar_feature)) + \
        #       (1 - self.theta) * (self.beta + 2 * self.theta) * np.matmul(c_s, np.transpose(c_s))
        a_1 = 2 * self.theta ** 2 * np.matmul(self.tar_feature, np.transpose(self.tar_feature)) + \
              self.beta ** 2 / 2 * np.matmul(self.src_feature, np.transpose(self.src_feature))
        print("a1 done")
        a_2 = self.beta * self.theta * (np.matmul(self.tar_feature, np.transpose(self.tar_feature))
                                        + np.matmul(self.src_feature, np.transpose(self.src_feature)))
        print("a2 done")
        a_3 = self.beta * self.theta * (np.matmul(self.tar_feature, np.transpose(self.tar_feature))
                                        + np.matmul(self.src_feature, np.transpose(self.src_feature)))
        print("a3 done")
        a_4 = 2 * self.theta ** 2 * np.matmul(self.src_feature, np.transpose(self.src_feature)) + \
              self.beta ** 2 / 2 * np.matmul(self.tar_feature, np.transpose(self.tar_feature))
        print("a4 done")
        a_upper = np.concatenate((a_1, a_2), axis=1)
        a_lower = np.concatenate((a_3, a_4), axis=1)
        a = np.concatenate((a_upper, a_lower), axis=0)
        print("a done")
        return a

    def calculate_eigenvalue(self):
        a = self.a
        print("in")
        U, _, _ = svd(a)
        print("U done")
        return U[:, :self.topk]

    def make_projected_data(self):
        U = self.U
        bt = U[:len(U)//2]
        bs = U[len(U)//2:]

        values=np.concatenate((bs, self.src_label), axis=1)
        valuet=np.concatenate((bt, self.tar_label), axis=1)

        src_projected = pd.DataFrame(values,index=self.src_index)
        tar_projected = pd.DataFrame(valuet,index=self.tar_index)

        return src_projected, tar_projected

if __name__ == '__main__':
    tar=pd.read_csv("./regeneration/sorted/targetRegenerationPCA.csv",index_col=0)
    src=pd.read_csv("./regeneration/sorted/sourceRegenerationPCA.csv",index_col=0)
    # tarNum,srcNum=tar.index.tolist(),src.index.tolist()
    # random.shuffle(srcNum)
    # random.shuffle(tarNum)
    # tarRandom,srcRandom=tar.iloc[tarNum],src.iloc[srcNum]
    arg=[0.4,12]
    p1=HEMAP(src,tar,arg)
    print(p1.src_projected)
    print(p1.tar_projected)
    p1.src_projected.to_csv("./projectedData/sorted/b0.4/12/srcProjected.csv",index=True)
    p1.tar_projected.to_csv("./projectedData/sorted/b0.4/12/tarProjected.csv", index=True)