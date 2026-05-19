nDataSet = 3

BATCH_SIZE = 32
epochs = 100
lr = 0.0003             #0.001
CLASS_NUM = 3
nBand = 198
HalfWidth = 5        #5
train_num = 40

momentum = 0.9
patch_size = 2 * HalfWidth + 1
no_cuda =False
cuda_id = '0'
l2_decay = 5e-4

seeds = [1328, 1372, 1417, 1421, 1535, 1535, 1588, 1610, 1631, 1670]#最后一个

train_end = 0
test_end = 0