nDataSet = 3

BATCH_SIZE = 32
epochs = 100
lr = 0.01
CLASS_NUM = 7
nBand = 48
HalfWidth =3
train_num = 60

momentum = 0.9
patch_size = 2 * HalfWidth + 1
no_cuda =False
cuda_id = '0'
l2_decay = 1e-6

seeds = [1372, 1328, 1417, 1421, 1535, 1535, 1588, 1610, 1631, 1670]

train_end = 0
test_end = 0

GIN_ch=24
temp=0.7