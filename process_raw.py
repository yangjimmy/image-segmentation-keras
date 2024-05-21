import numpy as np

raw_file = np.fromfile("E:\\YKL\\Thorlabs VSCAN Labeling\\raw\\VSCAN_0027.raw")
print(raw_file.shape)
raw_file.reshape([1024,640,3])