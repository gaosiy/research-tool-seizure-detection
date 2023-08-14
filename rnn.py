#BMINT AI-APIs
#Author： YanLi@Fudan university
#RNN
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter,sosfiltfilt, detrend
from sklearn.preprocessing import scale
from joblib import load
import torch
from torch import nn
import time
import numpy as np
import onnxruntime
import os
import ctypes
libc = ctypes.CDLL('libc.so.6')
#定义RNN架构
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=1,  
            hidden_size=64, 
            num_layers=1,  
            batch_first=True, 
        )
        self.flag = 1
        self.out = nn.Linear(64, 2) 

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  
        out = self.out(h_n[0])
        return out


print("using RNN model")
onnx_file = 'lstm.onnx'
ort_sess = onnxruntime.InferenceSession(onnx_file)

file_save = open("raw.txt",'w')
file_save_open = open("state.txt",'w')

fs = 2000.0
N = 2000
data = [0]*N
idx = [0]
flag = [0]
step = 2000

def recieve_data(sample):
    file_save.write(str(round(sample[0],1))+"\t"+str(round(sample[1],1))+"\n")
    if flag[0] == 0:
        file_save_open.write("0\n")
    else:
        file_save_open.write("1\n")
    if idx[0] < step:
        data.pop(0)
        data.append(round(sample[0],3))
        idx[0] += 1
        return [[0,0,1.1,0, 0]]

    
    x = sc.transform(data.reshape(1,-1))
    tem = x.reshape((1, 2000, 1))
    tem = torch.from_numpy(tem).to(torch.float32)
    onnx_data = np.array(tem).astype('float32')
    ort_inputs = {ort_sess.get_inputs()[0].name: onnx_data}
    pred = ort_sess.run(None, ort_inputs)

    idx[0] = 0
    for i in range(1):
        if pred[0][0][0] < pred[0][0][1]:
            if last_stim[0] == 0:
                output = [[i, 0.2, 0.1, 100, 1]]
                last_stim[0] = 1
            else:
                output = [[0, 0, 0, 0, 0]]
            flag[0] = 1
            print("seizure")

        else:
            output = [[i, 0, 0.12, 130, 1]]
            flag[0] = 0
            last_stim[0] = 0
            print("ns")
    return output

