nDataSet = 3

BATCH_SIZE = 32
epochs = 100
lr = 0.001
CLASS_NUM = 7
nBand = 102
HalfWidth = 5   #4  5
train_num = 20

momentum = 0.9
patch_size = 2 * HalfWidth + 1
no_cuda =False
cuda_id = '0'
l2_decay = 5e-5

seeds = [1631, 1610, 1588, 1670, 1328, 1328, 1417, 1421, 1535, 1535]#第四、六个

train_end = 0
test_end = 0