import time
start_time = time.time()
import h5py
import numpy as np
import torch

file_path = './data/train_shapes.h5'
f = h5py.File(file_path, 'r')
data = np.array(f['data'])
labels = np.array(f['label'])
print("cpu load time",time.time() - start_time)
val = torch.from_numpy(data).cuda()
print("gpu load time",time.time() - start_time)
