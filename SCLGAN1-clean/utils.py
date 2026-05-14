import os
import torch
import numpy as np
import scipy as sp
import scipy.stats
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from operator import truediv
import torch.utils.data as data
import sklearn.model_selection

def cdd(output_t1,output_t2):
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss
class Domain_Occ_loss(nn.Module):
    def __init__(self):
        super(Domain_Occ_loss,self).__init__()

    def forward(self,p1,p2):#(64,1)  (64,1)


        loss = - torch.mean(torch.log(p1 + 1e-6))
        loss -= torch.mean(torch.log(p2 + 1e-6))

        return loss
#load data methods
def cubeData(file_path):
    total = sio.loadmat(file_path)

    data1 = total['DataCube1'] #up
    data2 = total['DataCube2'] #pc
    gt1 = total['gt1']
    gt2 = total['gt2']

    # Data_Band_Scaler_s = data1
    # Data_Band_Scaler_t = data2
    # print('max and min ')

    # 归一化 [-0.5,0.5]
    # data1 = data1.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler_s = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))# - 0.5
    #
    # data2 = data2.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler_t = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2)) #- 0.5

    # # # 标准化
    data_s = data1.reshape(np.prod(data1.shape[:2]), np.prod(data1.shape[2:]))  # (111104,204)
    data_scaler_s = preprocessing.scale(data_s)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_s = data_scaler_s.reshape(data1.shape[0], data1.shape[1],data1.shape[2])

    data_t = data2.reshape(np.prod(data2.shape[:2]), np.prod(data2.shape[2:]))  # (111104,204)
    data_scaler_t = preprocessing.scale(data_t)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_t = data_scaler_t.reshape(data2.shape[0], data2.shape[1],data2.shape[2])
    print(np.max(Data_Band_Scaler_s),np.min(Data_Band_Scaler_s))
    print(np.max(Data_Band_Scaler_t),np.min(Data_Band_Scaler_t))
    return Data_Band_Scaler_s,Data_Band_Scaler_t, gt1,gt2  # image:(512,217,3),label:(512,217)

def indiana_cubeData(file_path):
    total = sio.loadmat(file_path)

    data1 = total['DataCube1']
    data2 = total['DataCube2']
    gt1 = total['gt1']
    gt2 = total['gt2']

    print(total.keys())

    data_s = data1.reshape(np.prod(data1.shape[:2]), np.prod(data1.shape[2:]))  # (111104,204)
    data_scaler_s = preprocessing.scale(data_s)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_s = data_scaler_s.reshape(data1.shape[0], data1.shape[1],data1.shape[2])
    data_t = data2.reshape(np.prod(data2.shape[:2]), np.prod(data2.shape[2:]))  # (111104,204)
    data_scaler_t = preprocessing.scale(data_t)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_t = data_scaler_t.reshape(data2.shape[0], data2.shape[1],data2.shape[2])

    return Data_Band_Scaler_s,Data_Band_Scaler_t, gt1,gt2

def load_data_houston(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['ori_data']

    GroundTruth = label_data['map']

    Data_Band_Scaler = data_all


    # # 归一化
    # data = data.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

def load_data_hyrank(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['ori_data']

    GroundTruth = label_data['map']

    # Data_Band_Scaler = data_all


    # # 归一化
    # data_all = data_all.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

def load_data_pavia(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)


    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    # data_all = data_all.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler = (data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all))

    # Data_Band_Scaler = data_all

    print(np.max(Data_Band_Scaler),np.min(Data_Band_Scaler))

    return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)

def get_sample_data(Sample_data, Sample_label, HalfWidth, num_per_class):
    print('get_sample_data() run...')
    print('The original sample data shape:',Sample_data.shape)
    print('The original sample label shape:', Sample_label.shape)
    print('HalfWidth:', HalfWidth)
    print('num_per_class:', num_per_class)

    nBand = Sample_data.shape[2]

    data = np.pad(Sample_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    print("data.shape",data.shape)
    label = np.pad(Sample_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    m = int(np.max(label))
    print(f'num_class : {m}')

    val = {}
    val_indices = []

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices[:num_per_class]
        val[i] = indices[num_per_class:]

    for i in range(m):
        train_indices += train[i]
        val_indices += val[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    #val
    print('the number of val data:', len(val_indices))#1270
    nVAL = len(val_indices)
    val_data = np.zeros([nVAL, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    val_label = np.zeros([nVAL], dtype=np.int64)
    RandPerm = val_indices
    RandPerm = np.array(RandPerm)

    for i in range(nVAL):
        val_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :],
                                                  (2, 0, 1))
        val_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    val_label = val_label - 1

    #train
    print('the number of processed data:', len(train_indices))#1260
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    processed_data = np.zeros([nTrain, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)#(1260, 48, 7, 7)
    print('sample label shape', processed_label.shape)#1260
    print('get_sample_data() end...')
    return processed_data, processed_label#, val_data, val_label

def get_all_data(All_data, All_label, HalfWidth):
    print('get_all_data() run...')
    print('The original data shape:', All_data.shape)
    nBand = All_data.shape[2]

    data = np.pad(All_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(All_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    num_class = int(np.max(label))
    print(f'num_class : {num_class}')

    for i in range(num_class):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if
                   label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices

    for i in range(num_class):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of all data:', len(train_indices))
    nTest = len(train_indices)
    index = np.zeros([nTest], dtype=np.int64)
    processed_data = np.zeros([nTest, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTest], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTest):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('processed all data shape:', processed_data.shape)
    print('processed all label shape:', processed_label.shape)
    print('get_all_data() end...')
    return index, processed_data, processed_label, label, RandPerm, Row, Column

def obtain_label(loader, net):
    start_test = True
    #print(net)
    net.eval()
    predict = np.array([], dtype=np.int64)

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            #print("data", data)
            inputs = data[0]
            labels = data[1]
            print("inputs的形状", inputs.shape)#([32, 48, 7, 7])
            print("labels的形状", labels.shape)#32
            inputs = inputs.cuda()
            feas, _, _, outputs, _ = net(inputs)
            print(feas.shape,outputs.shape)#torch.Size([32, 288]) torch.Size([32, 7])


            if start_test:
                all_fea = feas.float().cpu()

                all_output = outputs.float().cpu()
                print("if中的all_fea.shape,all_output.shape",all_fea.shape, all_output.shape)#if中的all_fea.shape,all_output.shape torch.Size([32, 288]) torch.Size([32, 7])
                all_label = labels.float()
                print("if all_label.shape",all_label.shape)#32
                print("if all_label",all_label)#i([5., 5., 4., 5., 5., 1., 4., 5., 1., 5., 5., 5., 5., 5., 5., 1., 5., 2.,2., 6., 5., 1., 5., 6., 5., 6., 5., 4., 5., 1., 4., 6.])
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)  # (53200,128)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)  # (53200,7)
                #print("else中的all_output.shape", all_output.shape)
                all_label = torch.cat((all_label, labels.float()), 0)  # 53200
                print("else all_label.shape", all_label.shape)
                print("else all_label", all_label)

    all_output = nn.Softmax(dim=1)(all_output)
    print("all_output.shape", all_output.shape)  # (53200,7)
    print("******")
    print("all_output", all_output)
    print("************")
    output, pred_label = torch.max(all_output, 1)#出错，all_output.shape为 torch.Size([53200, 7])
    print("pred_label.shape",pred_label.shape)
    print("pred_label",pred_label)
    print("***********")
    print(" output的形状",  output.shape)
    print("output",  output)
    print("************")
    predict = np.append(predict, pred_label.cpu().numpy())

    return predict, output

def Weighted_CrossEntropy(input_,labels):
    input_s = F.softmax(input_)
    entropy = -input_s * torch.log(input_s + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = weight / torch.sum(weight).detach().item()
    #print("cross:",nn.CrossEntropyLoss(reduction='none')(input_, labels))
    return torch.mean(weight * nn.CrossEntropyLoss(reduction='none')(input_, labels))

def twist_loss(p1,p2,alpha=1,beta=1):
    eps=1e-7 #ensure calculate
    #eps=0
    kl_div=((p2*p2.log()).sum(dim=1)-(p2*p1.log()).sum(dim=1)).mean()
    mean_entropy=-(p1*(p1.log()+eps)).sum(dim=1).mean()
    mean_prob=p1.mean(dim=0)
    entropy_mean=-(mean_prob*(mean_prob.log()+eps)).sum()

    return kl_div + alpha * mean_entropy - beta * entropy_mean
def extract_embeddings(model, dataloader):
    model.eval()
    n_samples = dataloader.batch_size * len(dataloader)
    embeddings = np.zeros((n_samples, model.n_outputs))
    labels = np.zeros(n_samples)
    k = 0

    for images, target in dataloader:
        with torch.no_grad():
            images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    """
    print("embeddings[0:k].shape", embeddings[0:k].shape)
    print("embeddings[0:k]", embeddings[0:k])
    print("**************")
    print("labels[0:k].shape", labels[0:k].shape)
    print("labels[0:k]", labels[0:k])
    """
    return embeddings[0:k], labels[0:k]

#data augmentation
def radiation_noise(data, alpha_range=(0.9, 1.1), beta=0.04): #pavia/houston = 0.04
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    #alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
    #noise_tensor = torch.tensor(noise, dtype=torch.float32)
    #alpha_tensor = alpha_tensor.to('cuda')
    #noise_tensor = noise_tensor.to('cuda')
    x = alpha * data + beta * noise
    return x

def radiation_noise_01(data, alpha_range=(0.9, 1.1), beta=0.04): #pavia/houston = 0.04
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
    noise_tensor = torch.tensor(noise, dtype=torch.float32)
    alpha_tensor = alpha_tensor.to('cuda')
    noise_tensor = noise_tensor.to('cuda')
    x = alpha_tensor * data + beta * noise_tensor
    return x
def flip_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5  # True
    vertical = np.random.random() > 0.5  # False
    if horizontal:
        #data = data.detach().to(device='cpu')
        data = np.fliplr(data)
        data = torch.from_numpy(data.copy())
    if vertical:
        #data = data.detach().to(device='cpu')
        data = np.flipud(data)
        data = torch.from_numpy(data.copy())
    return data


def flip_augmentation_01(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5  # True
    vertical = np.random.random() > 0.5  # False
    if horizontal:
        data = data.detach().to(device='cpu')
        data = np.fliplr(data)
        data = torch.from_numpy(data.copy())
    if vertical:
        data = data.detach().to(device='cpu')
        data = np.flipud(data)
        data = torch.from_numpy(data.copy())
    return data
#set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def Weighted_CrossEntropy(input_,labels):
    input_s = F.softmax(input_, dim=1)
    entropy = -input_s * torch.log(input_s + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = weight / torch.sum(weight).detach().item()
    #print("cross:",nn.CrossEntropyLoss(reduction='none')(input_, labels))
    return torch.mean(weight * nn.CrossEntropyLoss(reduction='none')(input_, labels))

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)

    fig.savefig(savePath, dpi = dpi)

    return 0

def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    train_label = []
    test_label = []
    if mode == 'random':
        if train_size == 1:
            random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)]
            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt = []
            test_set = []
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=23)
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            [test_label.append(i) for i in gt[tuple(test_indices)]]
            test_set = np.column_stack((test_indices[0],test_indices[1],test_label))

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt, train_set, test_set

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad