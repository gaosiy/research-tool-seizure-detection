#BMINT AI-APIs
#Author： YanLi@Fudan university
#SVM
import numpy as np
from sklearn import svm
from joblib import load
from scipy.signal import welch,sosfiltfilt,butter,detrend
import time
from sklearn.preprocessing import scale
import math

print(" using SVM model")
file_save = open("raw.txt",'w')
file_save_state = open("state.txt",'w')
fs = 2000.0
highcut = 45.0
order = 10
sos = butter(order, highcut, fs=fs, btype='lowpass',output='sos')

N = 2000
data = [0]*N
idx = [0]
flag = [0]
step = 2000

clf = load('svm.joblib')


# 特征提取，四个时域特征，方差，能量，非线性能量，香农熵
def shannon_entropy(X):
    n = len(X)
    # 使用Sturges规则计算直方图的bins数
    k = int(math.ceil(1 + math.log2(n)))
    # 计算每个bin中数据的数量
    hist, _ = np.histogram(X, bins=k)
    # 计算每个bin中数据的概率
    probs = hist / float(n)
    # 计算香农熵
    entropy = -np.sum([p * np.log2(p) for p in probs if p != 0])
    return entropy

def feature_extraction(X_A):
    var_A = np.var(X_A)
    energy_A = np.sum(np.square(X_A))
    nonlinear_energy_A = np.sum(X_A[1:-1]**2 - X_A[:-2]*X_A[2:])
    entropy_A = shannon_entropy(X_A)
    X_features = np.array([var_A, energy_A, nonlinear_energy_A, entropy_A])
    return X_features

def recieve_data(sample):
    
    file_save.write(str(round(sample[0],1))+"\t"+str(round(sample[1],1))+"\n")
    if flag[0] == 0:
        file_save_state.write("0\n")
    else:
        file_save_state.write("1\n")
    if idx[0] < step:
        data.pop(0)
        data.append(sample[0])
        idx[0] += 1
        return [[0,0,1.1,0, 0]]

    time_start = time.time()
    data_d = scale(data,with_std=False)
    data_f = sosfiltfilt(sos,data_d)

    data_feat = feature_extraction(data_f)
    pred = clf.predict(data_feat.reshape(1,-1))

    idx[0] = 0
    for i in range(1):
        if pred > 0:
            output = [[i, 0.1, 0.1, 130, 1]]
            flag[0] = 1
            print("seizure")
        else:
            output = [[i, 0, 0.1, 130, 1]]
            flag[0] = 0
            print("ns")
    return output
