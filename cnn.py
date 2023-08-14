#BMINT AI-APIs
#Author： YanLi@Fudan university
#CNN
import time
import numpy as np
from scipy import signal
import onnxruntime as rt
from sklearn.preprocessing import scale
from scipy.signal import welch,butter,filtfilt

file_save = open("raw.txt",'w')
file_save_open = open("state.txt",'w')
model = "lenet5.onnx"
sess = rt.InferenceSession(model)
input_name1 = sess.get_inputs()[0].name

print("using CNN model")

fs = 2000
flag = [0]
step = 2000
last_stim = [0]

N = 2000
data = [0]*N
idx = [0]


def recieve_data(sample):
    file_save.write(str(round(sample[0],1))+"\t"+str(round(sample[1],1))+"\n")
    if flag[0] == 0:
        file_save_open.write("0\n")
    else:
        file_save_open.write("1\n")
    if idx[0] < step:
        data.pop(0)
        data.append(sample[1])
        idx[0] += 1
        return [[0,0,1.1,0, 0]]

    f, t, Zxx = signal.stft(data, fs=2000, nperseg=1000, noverlap=905, padded=False)
    tem = np.abs(Zxx[(1< f)&(f<46)])
    tem = tem.repeat(2, axis = 0).repeat(2, axis = 1)
    tem = tem.reshape((1,44,44))

    input_image = np.array(tem).reshape(1, 1, 44, 44).astype('float32')
    new = np.ascontiguousarray(input_image,dtype=np.float32)
    pred = sess.run(None,{input_name1: new})
   
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

