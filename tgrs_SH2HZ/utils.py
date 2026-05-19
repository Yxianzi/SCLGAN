import math
import os
import torch
import numpy as np
from scipy.fft import fft2, ifft2
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import os
os.environ['MPLBACKEND'] = 'Agg'  # 在导入 matplotlib 前设置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import h5py
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
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

def Inaiana_cubeData(file_path):
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
def load_data_Salinas(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['salinas']

    GroundTruth = label_data['salinas_gt']

    Data_Band_Scaler = data_all


    # # 归一化
    # data = data.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)
def load_data_SalinasA(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['salinasA']

    GroundTruth = label_data['salinasA_gt']

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






def C2Seg_AB01(image_file, label_file):
    # 尝试用 scipy 读取，如果失败则使用 h5py
    try:
        # 尝试用 scipy 读取（适用于 v7.2 及以下版本）
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
    except NotImplementedError:
        # 使用 h5py 读取 v7.3 格式文件
        image_data = load_mat_v73(image_file)
        label_data = load_mat_v73(label_file)

    # 调试：查看文件中的键名
    print("图像文件键名:", list(image_data.keys()) if isinstance(image_data, dict) else "Not a dict")
    print("标签文件键名:", list(label_data.keys()) if isinstance(label_data, dict) else "Not a dict")

    # 根据实际情况提取数据
    # 如果是 h5py 读取的，可能需要不同的处理方式
    if isinstance(image_data, dict):
        # 原来的处理方式（scipy 读取）
        data_all = image_data['ori_data']
        GroundTruth = label_data['map']
    else:
        # h5py 读取的处理方式
        # 假设数据直接就是数组
        data_all = image_data
        GroundTruth = label_data

    print(f"数据形状: {data_all.shape}")
    print(f"标签形状: {GroundTruth.shape}")

    # 重塑数据为 2D 格式进行标准化
    original_shape = data_all.shape
    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))

    # 标准化 (X - X_mean) / X_std
    data_scaler = preprocessing.scale(data)

    # 重塑回原始形状
    Data_Band_Scaler = data_scaler.reshape(original_shape)

    print(f"标准化后 - 最大值: {np.max(Data_Band_Scaler):.4f}, 最小值: {np.min(Data_Band_Scaler):.4f}")

    return Data_Band_Scaler, GroundTruth


def load_mat_v73(file_path):
    """
    读取 Matlab v7.3 格式的 .mat 文件
    """
    f = h5py.File(file_path, 'r')

    # 打印文件结构以便调试
    print(f"文件 {file_path} 的结构:")
    print_keys(f)

    # 尝试查找常见的数据键名
    common_keys = ['ori_data', 'map', 'data', 'image', 'cube', 'hyperimg']

    for key in common_keys:
        if key in f:
            data = np.array(f[key])
            print(f"找到键 '{key}', 形状: {data.shape}")
            f.close()
            return data

    # 如果没有找到常见键，获取第一个数据集
    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            data = np.array(f[key])
            print(f"使用键 '{key}', 形状: {data.shape}")
            f.close()

            # Matlab 数据可能需要转置
            # 因为 Matlab 使用列主序，而 Python 使用行主序
            if len(data.shape) >= 2:
                # 对于高光谱数据，通常需要转置最后两个维度
                data = data.transpose()

            return data

    f.close()
    raise ValueError(f"在文件 {file_path} 中找不到有效的数据集")


def print_keys(obj, indent=0):
    """
    递归打印 h5py 文件结构
    """
    prefix = "  " * indent
    for key in obj.keys():
        item = obj[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}组: {key}")
            print_keys(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            shape_str = f"形状: {item.shape}" if hasattr(item, 'shape') else "无形状"
            dtype_str = f"类型: {item.dtype}" if hasattr(item, 'dtype') else ""
            print(f"{prefix}数据集: {key} - {shape_str}, {dtype_str}")

# 如果你确定文件都是 v7.3 格式，也可以直接使用这个简化版本
def C2Seg_AB02(image_file, label_file):
    """
    简化版本，假设所有文件都是 Matlab v7.3 格式
    """
    # 使用 h5py 读取
    f_img = h5py.File(image_file, 'r')
    f_lbl = h5py.File(label_file, 'r')

    # 获取数据
    # 注意：h5py 读取的数据可能需要转置
    data_all = np.array(f_img['ori_data']).transpose()  # 转置以匹配 Matlab 的顺序
    GroundTruth = np.array(f_lbl['map']).transpose()

    f_img.close()
    f_lbl.close()

    print(f"数据原始形状: {data_all.shape}")
    print(f"标签原始形状: {GroundTruth.shape}")

    # 重塑数据
    original_shape = data_all.shape
    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))

    # 标准化
    data_scaler = preprocessing.scale(data)

    # 重塑回原始形状
    Data_Band_Scaler = data_scaler.reshape(original_shape)

    print(f"标准化后 - 最大值: {np.max(Data_Band_Scaler):.4f}, 最小值: {np.min(Data_Band_Scaler):.4f}")

    return Data_Band_Scaler, GroundTruth

def C2Seg_AB(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['ori_data']

    GroundTruth = label_data['map']

    Data_Band_Scaler = data_all


    # # 归一化
    # data_all = data_all.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    #data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    #data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    #Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

def C2Seg_AB03(image_file, label_file):
    """
    加载C2Seg数据集，支持HDF5格式（Matlab v7.3）

    参数:
        image_file: 图像文件路径
        label_file: 标签文件路径

    返回:
        Data_Band_Scaler: 标准化后的图像数据 (H, W, C)
        GroundTruth: 地面真值标签 (H, W)
    """
    # 使用 h5py 读取 HDF5 格式的 .mat 文件
    # 1. 读取图像数据
    with h5py.File(image_file, 'r') as f_img:
        # 打印图像文件结构用于调试
        print(f"图像文件 {image_file} 中的键: {list(f_img.keys())}")

        # 尝试查找图像数据，优先使用 'ori_data'
        if 'ori_data' in f_img:
            data_all = np.array(f_img['ori_data'][:])
            print(f"从键 'ori_data' 读取图像数据，原始形状: {data_all.shape}")
        else:
            # 如果 'ori_data' 不存在，尝试其他可能的键名
            possible_keys = ['data', 'img', 'image', 'cube', 'hyperimg']
            for key in possible_keys:
                if key in f_img:
                    data_all = np.array(f_img[key][:])
                    print(f"从键 '{key}' 读取图像数据，原始形状: {data_all.shape}")
                    break
            else:
                # 如果都没有找到，使用第一个数据集
                first_key = list(f_img.keys())[0]
                data_all = np.array(f_img[first_key][:])
                print(f"从键 '{first_key}' 读取图像数据，原始形状: {data_all.shape}")

    # 2. 读取标签数据
    with h5py.File(label_file, 'r') as f_lbl:
        # 打印标签文件结构用于调试
        print(f"标签文件 {label_file} 中的键: {list(f_lbl.keys())}")

        # 尝试查找标签数据，优先使用 'map'
        if 'map' in f_lbl:
            GroundTruth = np.array(f_lbl['map'][:])
            print(f"从键 'map' 读取标签数据，原始形状: {GroundTruth.shape}")
        else:
            # 如果 'map' 不存在，尝试其他可能的键名
            possible_keys = ['label', 'gt', 'groundtruth', 'truth']
            for key in possible_keys:
                if key in f_lbl:
                    GroundTruth = np.array(f_lbl[key][:])
                    print(f"从键 '{key}' 读取标签数据，原始形状: {GroundTruth.shape}")
                    break
            else:
                # 如果都没有找到，使用第一个数据集
                first_key = list(f_lbl.keys())[0]
                GroundTruth = np.array(f_lbl[first_key][:])
                print(f"从键 '{first_key}' 读取标签数据，原始形状: {GroundTruth.shape}")

    # 3. 调整数据维度（根据你提供的代码模式）
    # 检查图像数据是否需要转置
    if len(data_all.shape) == 3:
        # 如果是3D数据，检查维度顺序
        # HDF5 存储的 Matlab 数据通常需要转置
        if data_all.shape[0] < data_all.shape[2]:  # 第一个维度小于第三个维度
            print("检测到需要转置图像数据 (H, W, C 顺序调整)...")
            data_all = data_all.transpose(2, 1, 0)  # 调整为 (H, W, C)
            print(f"转置后图像形状: {data_all.shape}")
        elif data_all.shape[0] < data_all.shape[1]:  # 第一个维度小于第二个维度
            print("检测到需要转置图像数据 (2D 情况)...")
            data_all = data_all.transpose(1, 0, 2)  # 调整为 (H, W, C)
            print(f"转置后图像形状: {data_all.shape}")

    # 4. 调整标签维度
    if len(GroundTruth.shape) == 2:
        # 检查是否需要转置以匹配图像数据
        if GroundTruth.shape[0] > GroundTruth.shape[1]:
            print("检测到需要转置标签数据...")
            GroundTruth = GroundTruth.T  # 转置成 (H, W) 格式
            print(f"转置后标签形状: {GroundTruth.shape}")
        elif GroundTruth.shape[0] < GroundTruth.shape[1] and data_all.shape[1] == GroundTruth.shape[0]:
            # 如果标签宽度大于高度，但高度与图像高度匹配，可能需要转置
            if data_all.shape[0] == GroundTruth.shape[1] and data_all.shape[1] == GroundTruth.shape[0]:
                print("转置标签以匹配图像维度...")
                GroundTruth = GroundTruth.T
                print(f"转置后标签形状: {GroundTruth.shape}")

    # 5. 确保标签是整数类型
    GroundTruth = GroundTruth.astype(np.int64)

    # 6. 数据标准化（保持你原来的标准化逻辑）
    print(f"标准化前图像数据形状: {data_all.shape}")
    print(f"标签数据形状: {GroundTruth.shape}")

    # 重塑数据为 2D 进行标准化
    original_shape = data_all.shape
    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    print(f"重塑为2D后的形状: {data.shape}")

    # 标准化 (X - X_mean) / X_std
    data_scaler = preprocessing.scale(data)

    # 重塑回原始 3D 形状
    Data_Band_Scaler = data_scaler.reshape(original_shape)

    print(f"标准化后图像形状: {Data_Band_Scaler.shape}")
    print(f"标准化范围 - 最大值: {np.max(Data_Band_Scaler):.6f}, 最小值: {np.min(Data_Band_Scaler):.6f}")

    # 打印标签统计信息
    unique_labels = np.unique(GroundTruth)
    print(f"标签中的唯一值: {unique_labels}")
    print(f"标签值范围: [{np.min(GroundTruth)}, {np.max(GroundTruth)}]")

    # 检查标签是否与图像尺寸匹配
    if Data_Band_Scaler.shape[0] != GroundTruth.shape[0] or Data_Band_Scaler.shape[1] != GroundTruth.shape[1]:
        print(f"警告: 图像和标签尺寸不匹配!")
        print(f"图像尺寸: {Data_Band_Scaler.shape[:2]}, 标签尺寸: {GroundTruth.shape}")

        # 尝试调整标签尺寸以匹配图像
        if GroundTruth.shape[0] == Data_Band_Scaler.shape[1] and GroundTruth.shape[1] == Data_Band_Scaler.shape[0]:
            print("自动转置标签以匹配图像...")
            GroundTruth = GroundTruth.T
        elif GroundTruth.shape[0] == Data_Band_Scaler.shape[0] // 2 or GroundTruth.shape[1] == Data_Band_Scaler.shape[
            1] // 2:
            print("标签可能是下采样版本，考虑上采样...")

    return Data_Band_Scaler, GroundTruth


# 为不同数据集提供便捷函数
def load_augsburg(image_file, label_file):
    """
    专门加载 Augsburg 数据集的函数
    """
    print("加载 Augsburg 数据集...")
    Data_Band_Scaler, GroundTruth = C2Seg_AB(image_file, label_file)

    # Augsburg 特定的后处理
    rgb_bands = (43, 21, 11)  # RGB波段索引
    label_values = ["1", "2", "3", "4", "5", "6", "7"]
    ignored_labels = [0]

    return Data_Band_Scaler, GroundTruth, rgb_bands, label_values, ignored_labels


def load_berlin(image_file, label_file):
    """
    专门加载 Berlin 数据集的函数
    """
    print("加载 Berlin 数据集...")
    Data_Band_Scaler, GroundTruth = C2Seg_AB(image_file, label_file)

    # Berlin 特定的后处理
    rgb_bands = (43, 21, 11)  # RGB波段索引
    label_values = ["1", "2", "3", "4", "5", "6", "7"]
    ignored_labels = [0]

    return Data_Band_Scaler, GroundTruth, rgb_bands, label_values, ignored_labels

def load_data_houston13(image_file, train_label_file,test_label_file):
    image_data = sio.loadmat(image_file)
    train_label_data = sio.loadmat(train_label_file)
    test_label_data = sio.loadmat(test_label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['data']

    GroundTruth_train = train_label_data['mask_train']

    GroundTruth_test = test_label_data['mask_test']

    # Data_Band_Scaler = data_all


    # # 归一化
    # data_all = data_all.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth_train,GroundTruth_test # image:(512,217,3),label:(512,217)

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


def get_all_data01(All_data, All_label, HalfWidth):
    print('get_all_data() run...')
    print('The original data shape:', All_data.shape)

    nBand = All_data.shape[2]
    patch_size = 2 * HalfWidth + 1

    # 计算总内存需求
    n_samples = len(np.nonzero(All_label)[0])  # 估计样本数
    memory_gb = n_samples * nBand * patch_size * patch_size * 4 / (1024 ** 3)
    print(f'估计内存需求: {memory_gb:.2f} GB')

    if memory_gb > 4.0:  # 超过4GB使用内存映射
        return get_all_data_mmap(All_data, All_label, HalfWidth)
    else:
        return get_all_data_normal(All_data, All_label, HalfWidth)


def get_all_data_normal(All_data, All_label, HalfWidth):
    """普通版本，用于小数据"""
    print('使用普通模式...')

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

    # 创建普通数组
    processed_data = np.zeros([nTest, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTest], dtype=np.int64)

    RandPerm = np.array(train_indices)

    for i in range(nTest):
        row_idx = Row[RandPerm[i]]
        col_idx = Column[RandPerm[i]]
        processed_data[i, :, :, :] = np.transpose(
            data[row_idx - HalfWidth: row_idx + HalfWidth + 1,
            col_idx - HalfWidth: col_idx + HalfWidth + 1, :],
            (2, 0, 1)
        )
        processed_label[i] = label[row_idx, col_idx].astype(np.int64)

    processed_label = processed_label - 1

    print('processed all data shape:', processed_data.shape)
    print('processed all label shape:', processed_label.shape)
    print('get_all_data() end...')

    index = np.arange(nTest, dtype=np.int64)

    return index, processed_data, processed_label, label, RandPerm, Row, Column


def get_all_data_mmap(All_data, All_label, HalfWidth):
    """内存映射版本，用于大数据"""
    print('使用内存映射模式...')

    import tempfile
    import os
    import atexit
    import shutil

    nBand = All_data.shape[2]
    patch_size = 2 * HalfWidth + 1

    # 填充数据
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

    # 创建临时目录 - 使用用户指定目录或非C盘
    import platform
    system = platform.system()

    if system == 'Windows':
        # 在Windows上，尝试使用D盘或其他非系统盘
        temp_dir = None
        for drive in ['D:', 'E:', 'F:']:
            if os.path.exists(drive):
                temp_dir = os.path.join(drive, 'temp_hsi_data')
                os.makedirs(temp_dir, exist_ok=True)
                break

        if temp_dir is None:
            # 回退到系统临时目录
            temp_dir = tempfile.mkdtemp(prefix='hsidata_')
    else:
        # Linux/Mac使用系统临时目录
        temp_dir = tempfile.mkdtemp(prefix='hsidata_')

    print(f'创建临时目录: {temp_dir}')

    # 临时文件路径
    data_path = os.path.join(temp_dir, 'processed_data.dat')
    label_path = os.path.join(temp_dir, 'processed_label.dat')
    meta_path = os.path.join(temp_dir, 'metadata.npz')

    # 创建内存映射文件
    print('创建内存映射文件...')

    # 数据文件
    data_shape = (nTest, nBand, patch_size, patch_size)
    data_mmap = np.memmap(data_path, dtype=np.float32, mode='w+', shape=data_shape)

    # 标签文件
    label_shape = (nTest,)
    label_mmap = np.memmap(label_path, dtype=np.int64, mode='w+', shape=label_shape)

    # 填充数据
    RandPerm = np.array(train_indices)
    chunk_size = 10000  # 分批处理

    print('开始填充数据...')
    for i in range(0, nTest, chunk_size):
        end_idx = min(i + chunk_size, nTest)
        batch_indices = RandPerm[i:end_idx]

        for j, idx in enumerate(batch_indices):
            row_idx = Row[idx]
            col_idx = Column[idx]
            patch = data[row_idx - HalfWidth: row_idx + HalfWidth + 1,
                    col_idx - HalfWidth: col_idx + HalfWidth + 1, :]
            data_mmap[i + j, :, :, :] = np.transpose(patch, (2, 0, 1))
            label_mmap[i + j] = label[row_idx, col_idx].astype(np.int64)

        if (i // chunk_size) % 10 == 0:
            print(f'处理进度: {i + len(batch_indices)}/{nTest} ({100 * (i + len(batch_indices)) / nTest:.1f}%)')

    # 标签从1开始转为从0开始
    label_mmap[:] = label_mmap[:] - 1

    # 确保数据写入磁盘
    data_mmap.flush()
    label_mmap.flush()
    print('数据已写入磁盘')

    # 重新以只读模式打开
    data_mmap_ro = np.memmap(data_path, dtype=np.float32, mode='r', shape=data_shape)
    label_mmap_ro = np.memmap(label_path, dtype=np.int64, mode='r', shape=label_shape)

    # 保存元数据
    np.savez(meta_path,
             RandPerm=RandPerm,
             Row=Row,
             Column=Column,
             label_shape=label.shape)

    # 创建清理函数 - 使用弱引用避免循环引用
    import weakref

    class TempFileManager:
        def __init__(self, temp_dir):
            self.temp_dir = temp_dir
            self._finalizer = weakref.finalize(self, self._cleanup)

        def _cleanup(self):
            try:
                if os.path.exists(self.temp_dir):
                    print(f'清理临时目录: {self.temp_dir}')
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception as e:
                print(f'清理临时目录失败: {e}')

    # 创建管理器
    manager = TempFileManager(temp_dir)

    # 将管理器存储在全局变量中，确保不会被垃圾回收
    if not hasattr(get_all_data_mmap, '_temp_managers'):
        get_all_data_mmap._temp_managers = []
    get_all_data_mmap._temp_managers.append(manager)

    # 注册atexit清理
    def cleanup_all():
        if hasattr(get_all_data_mmap, '_temp_managers'):
            for manager in get_all_data_mmap._temp_managers:
                manager._cleanup()

    atexit.register(cleanup_all)

    print('processed all data shape:', data_mmap_ro.shape)
    print('processed all label shape:', label_mmap_ro.shape)
    print(f'临时文件占用空间: {os.path.getsize(data_path) + os.path.getsize(label_path):.2f} GB')
    print('get_all_data() end...')

    index = np.arange(nTest, dtype=np.int64)

    # 返回内存映射数组和清理函数
    return index, data_mmap_ro, label_mmap_ro, label, RandPerm, Row, Column
def obtain_label(loader, net):
    start_test = True
    net.eval()
    predict = np.array([], dtype=np.int64)

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]

            inputs = inputs.cuda()
            feas,_,outputs,_ = net(inputs,inputs,false='test')

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)  # (53200,128)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)  # (53200,7)
                all_label = torch.cat((all_label, labels.float()), 0)  # 53200
    all_output = nn.Softmax(dim=1)(all_output)
    output, pred_label = torch.max(all_output, 1)
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
            embeddings[k:k+len(images)] = model.get_embedding(images,images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)

    return embeddings[0:k], labels[0:k]

def extract_embeddings01(model, dataloader,data_loader_t):
    model.eval()
    n_samples = dataloader.batch_size * len(data_loader_t)
    embeddings = np.zeros((n_samples, model.n_outputs))
    labels = np.zeros(n_samples)
    k = 0
    len_target_loader = len(dataloader)
    iter_target = iter(dataloader)
    num_iter = len_target_loader
    source_data, source_label = next(iter_target)

    for images, target in data_loader_t:
        with torch.no_grad():
            images = images.cuda()
            source_data = source_data.cuda()
            embeddings[k:k+len(images)] = model.get_embedding01(source_data,images).data.cpu().numpy()
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

def seed_everything(seed,use_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    if use_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def radiation_noise_01(data, alpha_range=(0.9, 1.1), beta=0.04): #pavia/houston = 0.04
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
    noise_tensor = torch.tensor(noise, dtype=torch.float32)
    alpha_tensor = alpha_tensor.to('cuda')
    noise_tensor = noise_tensor.to('cuda')
    x = alpha_tensor * data + beta * noise_tensor
    return x
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

# 定义熵损失函数
class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, logits):
        # 计算预测的概率分布
        probabilities = F.softmax(logits, dim=1)
        # 计算熵
        entropy = -torch.sum(probabilities * torch.log(probabilities), dim=1)
        # 计算熵的平均值作为损失
        loss = entropy.mean()
        return loss
# 判别器损失
# 计算损失
bce_loss = nn.BCELoss().cuda()
adversarial_loss = bce_loss
def discriminator_loss(D, real_images, fake_images, labels_real, labels_fake):
    # 真实图像的对抗损失
    _,_,_,real_images = D(real_images.cuda())
    softmax_output = nn.Softmax(dim=1)(real_images).detach()
    _, pseudo_label_0 = torch.max(softmax_output, 1)
    labels_fake_0 = torch.zeros(pseudo_label_0.size(0), 1).cuda()

    real_loss = adversarial_loss(labels_fake_0.cuda(), labels_real.cuda())

    _,_,_, fake_images = D(fake_images.detach())
    softmax_output = nn.Softmax(dim=1)(fake_images).detach()
    _, pseudo_label_1 = torch.max(softmax_output, 1)
    labels_fake_1 = torch.zeros(pseudo_label_1.size(0), 1).cuda()
    # 生成图像的对抗损失
    fake_loss = adversarial_loss(labels_fake_1, labels_fake)

    # 总判别器损失
    total_loss_D = (real_loss + fake_loss) * 0.5
    return total_loss_D
cycle_consistency_loss = nn.L1Loss().cuda()
# 生成器损失
def generator_loss( real_images,  cycle_images, identity_images, lambda_cycle=0.1):
    # 循环一致性损失
    cycle_loss = cycle_consistency_loss(real_images.cuda(), cycle_images.cuda())
    identity_loss = cycle_consistency_loss(real_images.cuda(), identity_images.cuda())
    #互信息损失

    # 总生成器损失
    total_loss_G = lambda_cycle * (cycle_loss + identity_loss)
    #print(total_loss_G)
    return total_loss_G
#计算分类损失
def classification_loss(output, target):
    loss = F.cross_entropy(output, target)
    return loss
# 计算特征匹配损失
def feature_matching_loss(fake_features, real_features):
    loss = F.l1_loss(fake_features, real_features)
    return loss


# 定义对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        # 获取batch size
        batch_size = embeddings.size(1)

        # 将嵌入向量正则化
        embeddings = F.normalize(embeddings, dim=1)

        # 计算两两嵌入向量的点积，得到相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T)

        # 将对角线元素置为很小的负数，避免自比较
        mask = torch.eye(batch_size, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask.cuda(), float('-inf'))

        # 获取正样本的相似度，即相同标签的样本
        pos_mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).T)
        pos_similarity = torch.masked_select(similarity_matrix, pos_mask)

        # 获取负样本的相似度，即不同标签的样本
        neg_mask = ~pos_mask
        neg_similarity = torch.masked_select(similarity_matrix, neg_mask)

        # 计算对比损失
        pos_similarity = torch.exp(pos_similarity / self.temperature)
        neg_similarity = torch.exp(neg_similarity / self.temperature)

        # 计算损失
        pos_loss = -torch.log(pos_similarity.sum(0) / pos_similarity.size(0))
        neg_loss = -torch.log(neg_similarity.sum(0) / neg_similarity.size(0))

        loss = pos_loss + neg_loss
        return loss.mean()

    # 实例化对比损失函数
contrastive_loss = ContrastiveLoss().cuda()

class CustomLoss(nn.Module):
    def __init__(self, lambda_d=1.0, lambda_t=1.0):
        super(CustomLoss, self).__init__()
        self.lambda_d = lambda_d  # 判别器损失权重
        self.lambda_t = lambda_t  # Transformer损失权重

    def forward(self, D_output_source, D_output_target, T_output, target_labels):
        """
        :param D_output_source: 判别器对源域样本的输出，形状为 (N_source, 1)
        :param D_output_target: 判别器对目标域样本的输出，形状为 (N_target, 1)
        :param T_output: Transformer模型的输出，形状为 (N_target, num_classes)
        :param target_labels: 目标域样本的真实标签，形状为 (N_target,)
        :return: 综合损失
        """

        # 判别器损失
        # 使用二元交叉熵损失
        D_loss_source = F.binary_cross_entropy(D_output_source.to(torch.float32), torch.ones_like(D_output_source.to(torch.float32)))
        D_loss_target = F.binary_cross_entropy(D_output_target.to(torch.float32), torch.zeros_like(D_output_target.to(torch.float32)))
        L_D = D_loss_source + D_loss_target

        # Transformer损失
        T_loss = F.cross_entropy(T_output, target_labels)

        # 综合损失
        total_loss = self.lambda_d * L_D + self.lambda_t * T_loss
        return total_loss
# 初始化损失函数
loss_function = CustomLoss()

def compute_loss(mu, log_var, y_true):
    # 计算方差
    var = torch.exp(log_var)

    # 计算高斯似然损失
    likelihood_loss = 0.5 * torch.sum((y_true - mu) ** 2 / var + log_var, dim=-1)

    # 计算KL散度
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - var, dim=-1)

    # 计算总的损失
    loss = torch.mean(likelihood_loss + kl_divergence)

    return loss

def frequency_alignment_loss(source_features, target_features):
    # 将空间特征转换为频域
    source_freq = torch.fft.fft2(source_features, dim=(-2))
    target_freq = torch.fft.fft2(target_features, dim=(-2))

    # 计算频域特征之间的损失
    loss01 = F.l1_loss(torch.abs(source_freq), torch.abs(target_freq))
    loss02 = correlation_alignment_loss(source_freq, target_freq)
    loss = loss01+loss02

    return loss


def spectral_frequency_alignment_loss(source_spectral_features, target_spectral_features):
    # 将光谱特征转换为频域
    source_freq = torch.fft.fft(source_spectral_features, dim=-1)
    target_freq = torch.fft.fft(target_spectral_features, dim=-1)

    # 计算频域特征之间的损失
    loss = torch.nn.functional.l1_loss(torch.abs(source_freq), torch.abs(target_freq))
    return loss

def correlation_alignment_loss(source_features, target_features):
    # 计算源域和目标域的特征的相关性矩阵
    source_cov = torch.cov(source_features.t())
    target_cov = torch.cov(target_features.t())
    # 计算相关对齐损失
    loss = torch.mean((source_cov - target_cov) ** 2)
    return loss


# 转换为频域特征
def convert_to_frequency_domain(data):
    #print(data.shape)
    data = data.detach().cpu().numpy()
    freq_data = np.abs(fft2(data, axes=(0, 1)))  # 对每个样本进行FFT
    return freq_data


# 对源域和目标域进行对齐
def align_domains(source_features, target_features):
    # 将特征展平
    source_features_flat = source_features.reshape(source_features.shape[0], -1)
    target_features_flat = target_features.reshape(target_features.shape[0], -1)

    # 标准化
    scaler = StandardScaler()
    source_features_scaled = scaler.fit_transform(source_features_flat)
    target_features_scaled = scaler.transform(target_features_flat)

    # 使用PCA进行对齐
    pca = PCA(n_components=source_features_scaled.shape[0])
    source_features_aligned = pca.fit_transform(source_features_scaled)
    target_features_aligned = pca.transform(target_features_scaled)



    return source_features_aligned, target_features_aligned


# 可视化频域特征
def visualize_frequency_features(source_freq, target_freq):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Source Domain Frequency Features')
    plt.imshow(np.mean(source_freq, axis=0), cmap='gray')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title('Target Domain Frequency Features')
    plt.imshow(np.mean(target_freq, axis=0), cmap='gray')
    plt.colorbar()

    plt.show()

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=0.04): #pavia/houston = 0.04
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    x = alpha * data + beta * noise
    return alpha * data + beta * noise


def flip_augmentation(data, flip_prob=0.3):
    """改进后的平缓增强函数
    Args:
        data: 输入数据 (H, W, C) 或 (H, W)
        flip_prob: 单次翻转的总概率（原函数实际概率是0.75=1-0.5*0.5）
    Returns:
        增强后的数据（保持原始类型和形状）
    """
    # 类型转换保险
    if isinstance(data, torch.Tensor):
        data_np = data.numpy()
        return_tensor = True
    else:
        data_np = data.copy()
        return_tensor = False

    # 平缓增强策略
    if np.random.random() < flip_prob:  # 总概率控制
        mode = np.random.choice(['hflip', 'vflip', 'none'], p=[0.4, 0.4, 0.2])  # 40%水平/垂直，20%不操作

        if mode == 'hflip':
            data_np = np.fliplr(data_np)
        elif mode == 'vflip':
            data_np = np.flipud(data_np)
        # else: 20%概率保持原样

    # 可选：添加微小随机平移（更平缓的增强）
    if np.random.random() < 0.5:  # 50%概率执行
        shift_pixels = np.random.randint(0, 2)  # 最多平移1像素
        axis = np.random.choice([0, 1])  # 选择平移方向
        data_np = np.roll(data_np, shift=shift_pixels, axis=axis)

    return torch.from_numpy(data_np.copy()) if return_tensor else data_np

def wasserstein_distance(source, target, reg=1e-3, num_iter=100):
    """
    使用Sinkhorn算法近似计算Wasserstein距离
    """
    C = torch.cdist(source, target)
    a = torch.ones(source.shape[0]) / source.shape[0]
    b = torch.ones(target.shape[0]) / target.shape[0]

    for i in range(num_iter):
        u = a / torch.mm(C, b)
        v = b / torch.mm(C.t(), u)

    pi = torch.diag(u) @ C @ torch.diag(v)
    return torch.sum(pi * C)

class ExtendedWassersteinLossWithHighOrder(nn.Module):
    def __init__(self, num_orders=3, weights=None):
        super(ExtendedWassersteinLossWithHighOrder, self).__init__()
        self.num_orders = num_orders
        if weights is None:
            self.weights = torch.ones(num_orders) / num_orders
        else:
            self.weights = torch.tensor(weights)
        self.weights = nn.Parameter(self.weights)  # 使得权重可学习

    def get_order_k_features(self, features, k):
        """
        简单示例，根据不同的阶数k，对特征进行不同的处理（这里只是简单分组，可替换更复杂逻辑）
        比如，将特征维度划分为不同部分作为不同阶特征
        """
        feature_dim = features.size(1)
        part_dim = feature_dim // self.num_orders
        start_idx = k * part_dim
        end_idx = (k + 1) * part_dim if k < self.num_orders - 1 else feature_dim
        return features[:, start_idx:end_idx]

    def forward(self, source_features, target_features):
        total_loss = 0
        for k in range(self.num_orders):
            source_order_k_features = self.get_order_k_features(source_features, k)
            target_order_k_features = self.get_order_k_features(target_features, k)

            # 计算这一阶特征对应的Wasserstein距离近似
            source_mean = torch.mean(source_order_k_features, dim=0, keepdim=True)
            source_cov = torch.cov(source_order_k_features.t())
            target_mean = torch.mean(target_order_k_features, dim=0, keepdim=True)
            target_cov = torch.cov(target_order_k_features.t())

            # 使用Frobenius范数代替平方和
            mean_distance = torch.norm(source_mean - target_mean, p='fro')
            cov_distance = torch.norm(source_cov - target_cov, p='fro')

            # 加权损失
            order_k_loss = mean_distance + cov_distance
            total_loss += self.weights[k] * order_k_loss

        # 添加正则化项
        reg_loss = torch.sum(self.weights ** 2)
        total_loss += reg_loss * 0.01  # 假设正则化权重为0.01

        return total_loss
def compute_prototypes(support_features, support_labels, num_classes):
    prototypes = []
    for c in range(num_classes):
        # 使用布尔索引，确保 support_labels 的一维与 support_features 的第一维匹配
        class_mask = support_labels == c
        if not class_mask.any():
            # 如果当前类别在支持集中没有样本，则跳过
            continue
        class_features = support_features[class_mask]
        # 计算类别的原型（特征的平均值）
        prototype = class_features.mean(dim=0)
        prototypes.append(prototype)
    # 将原型列表转换为张量
    prototypes = torch.stack(prototypes)
    return prototypes

def load_data_YRD(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['HSI']

    GroundTruth = label_data['GT']

    Data_Band_Scaler = data_all


    # # 归一化
    # data = data.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

# 使用谱聚类构建图
def spectral_clustering(features, n_clusters=3):
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_init=10)
    clustering.fit(features)
    return clustering.affinity_matrix_
"""
# 域不变图构建模型
class DomainInvariantGraphModel(nn.Module):
    def __init__(self):
        super(DomainInvariantGraphModel, self).__init__()
        # 定义一个可学习的图结构
        self.graph_weight = nn.Parameter(torch.randn(32, 32))

    def forward(self, features):
        # 使用可学习的图结构对特征进行加权
        graph_weight_normalized = F.normalize(self.graph_weight, p=1, dim=1)
        weighted_features = torch.mm(graph_weight_normalized, features)
        return weighted_features
"""
# 损失函数，这里使用MSE作为示例
def loss_function01(weighted_features_source, weighted_features_target):
    return F.mse_loss(weighted_features_source, weighted_features_target)

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.alpha
        return grad_input, None

class DomainInvariantGraphModel(nn.Module):
    def __init__(self, num_nodes, num_features):
        super(DomainInvariantGraphModel, self).__init__()
        # 定义一个可学习的图结构
        self.graph_weight = nn.Parameter(torch.randn(num_nodes, num_features))
        # 域分类器
        self.domain_classifier = DomainClassifier(num_features)
        """
        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 假设两个域
        )
        """
        # 梯度反转层
        self.grl = GradientReversalLayer.apply

    def forward(self, features, alpha=1.0,episode=1000):
        # 使用可学习的图结构对特征进行加权
        graph_weight_normalized = F.normalize(self.graph_weight, p=1, dim=1)
        weighted_features = torch.mm(graph_weight_normalized, features)
        # 应用梯度反转层
        reverse_features = self.grl(weighted_features, alpha)
        # 域分类器的输出
        domain_output = self.domain_classifier(reverse_features,episode)
        return weighted_features, domain_output

class DomainClassifier(nn.Module):
    def __init__(self,num_features):# torch.Size([1, 64, 7, 3, 3])
        super(DomainClassifier, self).__init__() #
        self.layer = nn.Sequential(
            nn.Linear(num_features, 128), #nn.Linear(320, 512), nn.Linear(FEATURE_DIM*CLASS_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.domain = nn.Linear(128, 2) # 512

    def forward(self, x, iter_num):
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10,10000.0)
        x.register_hook(grl_hook(coeff))
        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y

def calc_coeff(iter_num, low, high, alpha, max_iter):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def adjust_alpha(domain_accuracy, alpha):
    if domain_accuracy > 0.7:  # 如果域分类准确率较高
        return alpha * 1.1  # 增加 alpha
    else:
        return alpha * 0.9  # 减少 alpha


def reconstruction_loss(generated_data, real_data, loss_type='l1'):
    """
    计算重建损失。

    参数:
        generated_data (torch.Tensor): 生成器生成的数据。
        real_data (torch.Tensor): 原始数据。
        loss_type (str): 损失类型，支持 'l1' 或 'l2'。

    返回:
        torch.Tensor: 重建损失值。
    """
    if loss_type == 'l1':
        loss_fn = nn.L1Loss()
    elif loss_type == 'l2':
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("不支持的损失类型。请选择 'l1' 或 'l2'。")

    return loss_fn(generated_data, real_data)


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, lambda_consistency=0.1, method='mmd'):
        super(FeatureConsistencyLoss, self).__init__()
        self.lambda_consistency = lambda_consistency
        self.method = method

    def forward(self, gate_output_low, gate_output_high, low_features, high_features):
        # 特征一致性损失
        consistency_loss = 0
        for i in range(low_features.size(0)):  # 遍历每个样本
            # 计算一致性区域的特征差异
            low_gated = low_features[i] * gate_output_low[i].expand_as(low_features[i])
            high_gated = high_features[i] * gate_output_high[i].expand_as(high_features[i])

            if self.method == 'cosine':
                # 使用余弦相似度
                diff = 1 - F.cosine_similarity(low_gated, high_gated, dim=-1).mean()
            elif self.method == 'mmd':
                # 使用 MMD 损失
                diff = mmd_loss(low_gated.unsqueeze(0), high_gated.unsqueeze(0))
            else:
                # 默认使用 L2 范数
                diff = torch.norm(low_gated - high_gated, p=2)

            consistency_loss += diff

        consistency_loss /= low_features.size(0)  # 取平均值

        # 总损失
        total_loss = self.lambda_consistency * consistency_loss
        return total_loss

def mmd_loss(x, y, kernel='rbf'):
    if kernel == 'rbf':
        # 使用高斯核计算 MMD
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())
        gamma = 1.0 / x.size(1)  # 带宽参数
        xx = torch.exp(-gamma * (xx - torch.diag(xx).unsqueeze(1)))
        yy = torch.exp(-gamma * (yy - torch.diag(yy).unsqueeze(1)))
        xy = torch.exp(-gamma * (xy - torch.diag(xy).unsqueeze(1)))
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


def align_clusters_by_centers(source_centers, target_centers):
    """通过聚类中心距离匹配源域和目标域的簇"""
    # 计算聚类中心之间的距离矩阵
    dist_matrix = pairwise_distances(source_centers, target_centers)
    # 使用匈牙利算法找到最优匹配
    _, col_ind = linear_sum_assignment(dist_matrix)
    return col_ind  # 返回目标域簇到源域簇的映射关系



def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor, y: torch.Tensor, label, name,
              source_color='r', target_color='b'):
    source_feature = source_feature.cpu().detach().numpy()
    target_feature = target_feature.cpu().detach().numpy()
    print(y.shape)
    if len(y.shape) > 1:
        _, y = torch.max(y, dim=1)
    y = y.cpu().detach().numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)
    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red', 4: 'purple', 5: 'brown', 6: 'cyan'}
    unique_classes = np.unique(y)

    for class_label in unique_classes:
        plt.figure(figsize=(10, 8))
        indices = np.where(y == class_label)
        X_class = X_tsne[indices]
        domain_class = domains[indices]
        # 绘制每个点
        for i in range(len(X_class)):
            if domain_class[i] == 1:
                plt.scatter(X_class[i, 0], X_class[i, 1],
                            color=colors[2],
                            marker='o',
                            alpha=0.7)
            else:
                # plt.scatter(X_class[i, 0], X_class[i, 1],
                #             edgecolors=colors[class_label],marker='o',c="none",
                #             alpha=0.7)
                plt.scatter(X_class[i, 0], X_class[i, 1],color=colors[1],marker='o',alpha=0.7)
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2],
                       label=f'Source - Class {class_label}', markersize=16),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1],
                       label=f'Target - Class {class_label}', markersize=16),
        ], loc='upper right', fontsize=15)
        # 添加标题和标签
        plt.title(f't-SNE Visualization for Class {class_label}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid()
        # 显示图形
        plt.show()
        plt.close()

    # 计算目标域聚类指标

    source_indices = np.where(domains == 1)[0]
    source_features = features[source_indices]
    source_labels = y[source_indices]

    target_indices = np.where(domains == 0)[0]
    target_features = features[target_indices]
    target_labels = y[target_indices]

    # 可视化
    #plt.figure(figsize=(15, 6))
    plt.figure(figsize=(10, 8))

    for i in range(len(X_tsne)):
        if label != None:
            if y[i].item() in label:
                if domains[i] == 1:
                    mask_source = (domains == 1) & (np.isin(y, label)) if label else (domains == 1)
                    plt.scatter(X_tsne[mask_source, 0], X_tsne[mask_source, 1],
                                c=[colors[l] for l in y[mask_source]],
                                marker='o', alpha=0.7)
                else:
                    mask_target = (domains == 0) & (np.isin(y, label)) if label else (domains == 0)
                    plt.scatter(X_tsne[mask_target, 0], X_tsne[mask_target, 1],
                                c=[colors[l] for l in y[mask_target]],  # 使用c参数设置填充色
                                marker='^',  # 三角形标记
                                alpha=0.7)
        else:
            # 对源域进行K-means聚类
            source_kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
            source_cluster_labels = source_kmeans.fit_predict(source_features)

            # 对目标域进行K-means聚类
            target_kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
            target_cluster_labels = target_kmeans.fit_predict(target_features)
            # 2. 获取聚类中心
            source_centers = source_kmeans.cluster_centers_
            target_centers = target_kmeans.cluster_centers_

            # 3. 对齐目标域簇编号
            mapping = align_clusters_by_centers(source_centers, target_centers)
            target_cluster_labels_aligned = np.array([mapping[l] for l in target_cluster_labels])

            for i in range(len(unique_classes)):
                mask = (source_cluster_labels == i)
                plt.scatter(X_tsne[source_indices][mask, 0], X_tsne[source_indices][mask, 1],
                            color=colors[i], marker='^', label=f'Source Class-{i}', alpha=0.7, s=100)

            for i in range(len(unique_classes)):
                #mask = (target_cluster_labels == i)
                mask = (target_cluster_labels_aligned == i)
                plt.scatter(X_tsne[target_indices][mask, 0], X_tsne[target_indices][mask, 1],
                            color=colors[i], marker='o', label=f'Target Class-{i}', alpha=0.7, s=100)



            # 绘制聚类中心
            #if len(kmeans.cluster_centers_) > 5:  # 只有中心点足够多时才使用t-SNE
                #centers_tsne = TSNE(n_components=2, perplexity=min(5, len(kmeans.cluster_centers_) - 1)).fit_transform(
                    #kmeans.cluster_centers_)
            #else:
                #centers_tsne = kmeans.cluster_centers_[:, :2]  # 取前两个维度
            #plt.scatter(centers_tsne[:, 0], centers_tsne[:, 1],c='black', marker='X', s=200, label='Centers')

            #plt.scatter(X_tsne[i, 0], X_tsne[i, 1],color=colors[y[i].item()],marker='o',alpha=0.7)
            #mask_target = (domains == 0) & (np.isin(y, label)) if label else (domains == 0)
            #plt.scatter(X_tsne[mask_target, 0], X_tsne[mask_target, 1],c=[colors[l] for l in y[mask_target]],marker='^',  alpha=0.7)
    #图例
    plt.legend(handles=[plt.Line2D([0], [0], marker='^', color='w',markerfacecolor=colors[i], label=f'Source - Class {i}', markersize=15) for i in unique_classes]+
                       [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                                   label=f'Target - Class {i}', markersize=15) for i in unique_classes],
               loc='upper right', fontsize=16)
    plt.title(f't-SNE Visualization for All Classes')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid()
    # 显示图形
    plt.show()
    plt.close()


def visualize1(source_feature: torch.Tensor, target_feature: torch.Tensor, y: torch.Tensor, label=None, name=None):
    # 数据准备
    source_feature = source_feature.cpu().detach().numpy()
    target_feature = target_feature.cpu().detach().numpy()

    if len(y.shape) > 1:
        _, y = torch.max(y, dim=1)
    y = y.cpu().detach().numpy()

    features = np.concatenate([source_feature, target_feature], axis=0)
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # t-SNE降维
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # 颜色和类别设置
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(y))))  # 自动生成颜色
    colors = {i: colors[i] for i in range(len(np.unique(y)))}
    unique_classes = np.unique(y)

    # 获取域索引
    source_indices = np.where(domains == 1)[0]
    target_indices = np.where(domains == 0)[0]

    # 情况1：有指定标签时，直接按标签可视化
    if label is not None:
        plt.figure(figsize=(10, 8))
        for cls in unique_classes:
            if cls not in label:
                continue

            # 源域
            mask = (y == cls) & (domains == 1)
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        color=colors[cls], marker='^', alpha=0.7, s=50,
                        label=f'Source-Class-{cls}')

            # 目标域
            mask = (y == cls) & (domains == 0)
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        color=colors[cls], marker='o', alpha=0.7, s=50,
                        label=f'Target-Class-{cls}')

    # 情况2：无指定标签时，进行聚类分析
    else:
        # 对源域和目标域分别聚类
        source_kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
        source_cluster_labels = source_kmeans.fit_predict(source_feature)

        target_kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
        target_cluster_labels = target_kmeans.fit_predict(target_feature)

        # 对齐聚类标签
        mapping = align_clusters_by_centers(source_kmeans.cluster_centers_,
                                            target_kmeans.cluster_centers_)
        target_cluster_labels_aligned = np.array([mapping[l] for l in target_cluster_labels])

        # 可视化聚类结果
        plt.figure(figsize=(10, 8))
        for i in range(len(unique_classes)):
            # 源域 (三角形)
            mask = (source_cluster_labels == i)
            plt.scatter(X_tsne[source_indices][mask, 0], X_tsne[source_indices][mask, 1],
                        color=colors[i], marker='^', alpha=0.7, s=100,
                        label=f'Source-Cluster-{i}' if i == 0 else "")

            # 目标域 (圆形)
            mask = (target_cluster_labels_aligned == i)
            plt.scatter(X_tsne[target_indices][mask, 0], X_tsne[target_indices][mask, 1],
                        color=colors[i], marker='o', alpha=0.7, s=100,
                        label=f'Target-Cluster-{i}' if i == 0 else "")

    # 统一图例样式
    legend_elements = [
                          plt.Line2D([], [], marker='^', color='gray', linestyle='None',
                                     markersize=10, label='Source'),
                          plt.Line2D([], [], marker='o', color='gray', linestyle='None',
                                     markersize=10, label='Target')
                      ] + [
                          plt.Line2D([], [], marker='s', color=colors[i], linestyle='None',
                                     markersize=10, label=f'Class-{i}')
                          for i in unique_classes
                      ]

    plt.legend(handles=legend_elements, loc='upper right')
    plt.title('t-SNE Visualization with Aligned Clustering')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()


def balanced_tsne_visualization(
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        y: torch.Tensor,
        label=None,
        n_samples=180,
        random_state=33
):
    # ============ 数据预处理 ============
    # 转换为numpy并确保标签为1D
    source_feature = source_feature.cpu().numpy()
    target_feature = target_feature.cpu().numpy()
    y = y.cpu().numpy()
    if y.ndim > 1:
        y = np.argmax(y, axis=1)

    # ============ 平衡采样 ============
    np.random.seed(random_state)
    # 源域采样
    source_idx = np.random.choice(
        len(source_feature),
        size=max(n_samples, len(source_feature)),
        replace=False
    )
    # 目标域采样
    target_idx = np.random.choice(
        len(target_feature),
        size=max(n_samples, len(target_feature)),
        replace=False
    )

    # 应用采样
    source_feature = source_feature[source_idx]
    target_feature = target_feature[target_idx]
    y_source = y[source_idx]
    y_target = y[target_idx]

    # ============ 数据合并 ============
    features = np.vstack([source_feature, target_feature])
    domains = np.concatenate([
        np.ones(len(source_feature)),  # 源域标记为1
        np.zeros(len(target_feature))  # 目标域标记为0
    ])
    y_combined = np.concatenate([y_source, y_target])
    unique_classes = np.unique(y_combined)

    # ============ t-SNE降维 ============
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(features) // 4),
        random_state=random_state
    )
    X_tsne = tsne.fit_transform(features)
    # 获取域索引
    source_indices = np.where(domains == 1)[0]
    target_indices = np.where(domains == 0)[0]
    # ============ 可视化设置 ============
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    # ============ 可视化逻辑 ============
    if label is not None:
        # 模式1：按真实标签可视化
        for cls in unique_classes:
            if cls not in label:
                continue

            # 绘制源域样本（三角形）
            mask = (y_combined == cls) & (domains == 1)
            plt.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                color=color_map[cls], marker='^', s=50,
                label=f'Source-Class-{cls}'
            )

            # 绘制目标域样本（圆形）
            mask = (y_combined == cls) & (domains == 0)
            plt.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                color=color_map[cls], marker='o', s=50,
                label=f'Target-Class-{cls}'
            )
    else:
        # 对源域和目标域分别聚类
        source_kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
        source_cluster_labels = source_kmeans.fit_predict(source_feature)

        target_kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
        target_cluster_labels = target_kmeans.fit_predict(target_feature)

        # 对齐聚类标签
        mapping = align_clusters_by_centers(source_kmeans.cluster_centers_,
                                            target_kmeans.cluster_centers_)
        target_cluster_labels_aligned = np.array([mapping[l] for l in target_cluster_labels])

        # 可视化聚类结果
        plt.figure(figsize=(10, 8))
        for i in range(len(unique_classes)):
            # 源域 (三角形)
            mask = (source_cluster_labels == i)
            plt.scatter(X_tsne[source_indices][mask, 0], X_tsne[source_indices][mask, 1],
                        color=colors[i], marker='^', alpha=0.7, s=100,
                        label=f'Source-Cluster-{i}' if i == 0 else "")

            # 目标域 (圆形)
            mask = (target_cluster_labels_aligned == i)
            plt.scatter(X_tsne[target_indices][mask, 0], X_tsne[target_indices][mask, 1],
                        color=colors[i], marker='o', alpha=0.7, s=100,
                        label=f'Target-Cluster-{i}' if i == 0 else "")

        # 统一图例样式
    legend_elements = [
                          plt.Line2D([], [], marker='^', color='gray', linestyle='None',
                                     markersize=10, label='Source'),
                          plt.Line2D([], [], marker='o', color='gray', linestyle='None',
                                     markersize=10, label='Target')
                      ] + [
                          plt.Line2D([], [], marker='s', color=colors[i], linestyle='None',
                                     markersize=10, label=f'Class-{i}')
                          for i in unique_classes
                      ]

    # 统一图例样式
    legend_elements = [
                          plt.Line2D([], [], marker='^', color='gray', linestyle='None',
                                     markersize=10, label='Source'),
                          plt.Line2D([], [], marker='o', color='gray', linestyle='None',
                                     markersize=10, label='Target')
                      ] + [
                          plt.Line2D([], [], marker='s', color=colors[i], linestyle='None',
                                     markersize=10, label=f'Class-{i}')
                          for i in unique_classes
                      ]

    plt.legend(handles=legend_elements, loc='upper right')
    plt.title('t-SNE Visualization with Aligned Clustering')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()


def visualize_with_clustering1(
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        y_source: torch.Tensor,
        y_target: torch.Tensor = None,  # 可选的目标域标签
        label=None,
        n_samples=200,
        random_state=33
):
    """
    完整版可视化函数（支持两种模式）
    模式1：当提供y_target时，直接使用目标域伪标签
    模式2：无y_target时，执行聚类分析
    """
    # ============ 数据预处理 ============
    source_feature = source_feature.cpu().numpy()
    target_feature = target_feature.cpu().numpy()
    y_source = y_source.cpu().numpy().flatten()

    # 平衡采样
    np.random.seed(random_state)
    source_idx = np.random.choice(len(source_feature),
                                  size=min(n_samples, len(source_feature)),
                                  replace=False)
    source_feature = source_feature[source_idx]
    y_source = y_source[source_idx]

    # ============ 目标域处理 ============
    if y_target is not None:
        # 模式1：使用目标域伪标签
        y_target = y_target.cpu().numpy().flatten()
        target_idx = np.random.choice(len(target_feature),
                                      size=min(n_samples, len(target_feature)),
                                      replace=False)
        target_feature = target_feature[target_idx]
        y_target = y_target[target_idx]
        use_true_labels = True
    else:
        # 模式2：聚类分析
        target_idx = np.random.choice(len(target_feature),
                                      size=min(n_samples, len(target_feature)),
                                      replace=False)
        target_feature = target_feature[target_idx]
        use_true_labels = False

    # ============ 数据合并 ============
    features = np.vstack([source_feature, target_feature])
    domains = np.concatenate([np.ones(len(source_feature)),
                              np.zeros(len(target_feature))])

    if use_true_labels:
        y_combined = np.concatenate([y_source, y_target])
    else:
        y_combined = np.concatenate([y_source, np.zeros(len(target_feature))])  # 目标域标签占位

    unique_classes = np.unique(y_source)  # 以源域类别为基准
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    # ============ t-SNE降维 ============
    X_tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(features) // 4),
        random_state=random_state
    ).fit_transform(features)

    # ============ 可视化 ============
    plt.figure(figsize=(10, 6))

    if use_true_labels:
        # 模式1：源域真实标签和目标域伪标签可视化
        for cls in unique_classes:
            if label is not None and cls not in label:
                continue

            # 源域
            mask = (y_combined == cls) & (domains == 1)
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        color=color_map[cls], marker='^', s=80,
                        label=f'Src-{cls}' if cls == unique_classes[0] else "")

            # 目标域
            mask = (y_combined == cls) & (domains == 0)
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        color=color_map[cls], marker='o', s=80,
                        label=f'Tgt-{cls}' if cls == unique_classes[0] else "")
    else:
        # 模式2：聚类分析（修正版）
        target_idx = np.random.choice(len(target_feature),
                                      size=min(n_samples, len(target_feature)),
                                      replace=False)
        target_feature = target_feature[target_idx]

        # ===== 关键修改1：特征归一化 =====

        scaler = StandardScaler()
        source_feature_scaled = scaler.fit_transform(source_feature)
        target_feature_scaled = scaler.transform(target_feature)

        # ===== 关键修改2：聚类与标签对齐 =====
        # 源域聚类（强制使用真实标签数量）
        source_kmeans = KMeans(n_clusters=len(np.unique(y_source)), random_state=random_state)
        source_labels = source_kmeans.fit_predict(source_feature_scaled)

        # 目标域聚类
        target_kmeans = KMeans(n_clusters=len(np.unique(y_source)), random_state=random_state)
        target_labels_raw = target_kmeans.fit_predict(target_feature_scaled)

        # 标签对齐（基于聚类中心余弦相似度）
        cost_matrix = 1 - cosine_similarity(source_kmeans.cluster_centers_,
                                            target_kmeans.cluster_centers_)



        _, col_ind = linear_sum_assignment(cost_matrix)
        target_labels_aligned = np.array([col_ind[l] for l in target_labels_raw])

        # ===== 关键修改3：颜色映射到源域真实标签 =====
        # 统计源域聚类标签与真实标签的映射关系
        conf_matrix = confusion_matrix(y_source, source_labels)
        source_label_mapping = np.argmax(conf_matrix, axis=0)

        # 应用映射关系
        source_labels_mapped = np.array([source_label_mapping[l] for l in source_labels])
        target_labels_mapped = np.array([source_label_mapping[l] for l in target_labels_aligned])

        # ============ 数据合并 ============
        features = np.vstack([source_feature_scaled, target_feature_scaled])
        domains = np.concatenate([np.ones(len(source_feature)),
                                  np.zeros(len(target_feature))])
        cluster_labels = np.concatenate([source_labels_mapped, target_labels_mapped])

        # ===== 可视化 =====
        unique_classes = np.unique(y_source)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

        plt.figure(figsize=(10, 6))
        for cls in unique_classes:
            if label is not None and cls not in label:
                continue

            # 源域
            mask = (cluster_labels == cls) & (domains == 1)
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        color=color_map[cls], marker='^', s=80,
                        label=f'Src-{cls}' if cls == unique_classes[0] else "")

            # 目标域
            mask = (cluster_labels == cls) & (domains == 0)
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        color=color_map[cls], marker='o', s=80,
                        label=f'Tgt-{cls}' if cls == unique_classes[0] else "")

    # ============ 图例和样式 ============
    # 统一图例样式
    legend_elements = [
                          plt.Line2D([], [], marker='^', color='gray', linestyle='None',
                                     markersize=10, label='Source'),
                          plt.Line2D([], [], marker='o', color='gray', linestyle='None',
                                     markersize=10, label='Target')
                      ] + [
                          plt.Line2D([], [], marker='s', color=colors[i], linestyle='None',
                                     markersize=10, label=f'Class-{i}')
                          for i in unique_classes
                      ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1))
    #title = 'With True Labels' if use_true_labels else 'With Clustering'
    #plt.title(f't-SNE Visualization ({title})\nSamples: {len(source_feature)} per domain')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # 新增：移除坐标轴刻度
    plt.xticks([])
    plt.yticks([])
    plt.savefig('YRD8.png', dpi=300, bbox_inches='tight')
    plt.show()









def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:

    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c

    where C is number of classes.

    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


def calc_align_weight_accuracy(cls_loss, src_logits, src_labels, min_weight=0.1, max_weight=1.0):
    """根据分类准确率调整权重"""
    with torch.no_grad():
        preds = torch.argmax(src_logits, dim=1)
        acc = (preds == src_labels).float().mean()  # 当前batch准确率
        weight = max_weight - (max_weight - min_weight) * acc  # 准确率越高，权重越低
    return weight

def calc_align_weight_ratio(cls_loss, align_loss, min_weight=0.1, max_weight=1.0, smooth=0.1):
    """基于损失比例的动态权重（平滑版）"""
    with torch.no_grad():
        ratio = (align_loss / cls_loss).clamp(min=1e-3, max=1e3)  # 防止除零或极端值
        log_ratio = torch.log(ratio)
        weight = min_weight + (max_weight - min_weight) * torch.sigmoid(log_ratio * smooth)
    return weight

def calc_align_weight_curriculum(epoch, max_epoch, min_weight=0.1, max_weight=1.0):
    """渐进式权重调整"""
    if epoch < max_epoch // 3:  # 前1/3训练阶段
        return min_weight
    elif epoch < 2 * max_epoch // 3:  # 中间阶段
        return (max_weight + min_weight) / 2
    else:  # 最后阶段
        return max_weight

def calc_align_weight_confidence(tgt_logits, threshold=0.9, min_weight=0.1, max_weight=1.0):
    """根据目标域预测置信度调整权重"""
    with torch.no_grad():
        probs = F.softmax(tgt_logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        high_conf = (max_probs > threshold).float().mean()  # 高置信度样本比例
        weight = min_weight + (max_weight - min_weight) * high_conf
    return weight


def calc_align_weight_safe(cls_loss, align_loss, model, epoch, src_logits=None, src_labels=None,
                           max_epoch=100, min_weight=0.1, max_weight=0.5, alpha=0.9):
    """
    综合安全权重策略（课程学习 + EMA平滑 + 异常值裁剪）

    参数:
        cls_loss: 分类损失值
        align_loss: 对齐损失值
        model: 当前模型
        epoch: 当前epoch数
        src_logits: 源域预测logits（用于准确率计算）
        src_labels: 源域真实标签
        max_epoch: 总训练epoch数
        min_weight: 最小权重值
        max_weight: 最大权重值
        alpha: EMA平滑系数
    """
    # 1. 课程学习基础权重（线性增长）
    curriculum_weight = min_weight + (max_weight - min_weight) * (epoch / max_epoch)

    # 2. 当前batch动态权重
    if src_logits is not None and src_labels is not None:
        with torch.no_grad():
            preds = torch.argmax(src_logits, dim=1)
            acc = (preds == src_labels).float().mean()
            current_weight = max_weight - (max_weight - min_weight) * acc
    else:
        ratio = (align_loss.detach() / (cls_loss.detach() + 1e-6)).clamp(0.1, 10)
        current_weight = min_weight + (max_weight - min_weight) * torch.sigmoid(torch.log(ratio))

    # 3. 混合策略：课程学习权重作为基础，动态权重作为调整
    mixed_weight = 0.5 * curriculum_weight + 0.5 * current_weight

    # 4. EMA平滑
    if not hasattr(model, 'running_weight'):
        model.running_weight = mixed_weight.item() if torch.is_tensor(mixed_weight) else mixed_weight
    model.running_weight = alpha * model.running_weight + (1 - alpha) * mixed_weight

    # 5. 安全裁剪
    safe_weight = np.clip(model.running_weight.cpu(), min_weight, max_weight)

    return safe_weight

def graph_data_augmentation(features, edge_index, drop_rate=0.1):
    # 随机删除边
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > drop_rate
    edge_index_aug = edge_index[:, mask]
    # 随机扰动节点特征
    noise = torch.randn_like(features) * 0.1
    features_aug = features + noise
    return features_aug, edge_index_aug

class NonTrainableDIGM(nn.Module):
    def __init__(self, num_nodes, num_features):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features

        # 固定全连接图结构（无需训练）
        self.edge_index = self._create_fully_connected_graph(num_nodes).cuda()

    def _create_fully_connected_graph(self, num_nodes):
        """生成全连接图的边索引"""
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def _compute_adaptive_weights(self, features):
        """基于特征相似度的自适应权重计算"""
        # 余弦相似度矩阵 [num_nodes, num_nodes]
        sim_matrix = F.cosine_similarity(
            features.unsqueeze(1),
            features.unsqueeze(0),
            dim=2
        )

        # 稀疏化处理：保留top-k连接
        k = int(self.num_nodes * 0.3)  # 保留30%的强连接
        topk_values, _ = torch.topk(sim_matrix, k=k, dim=1)
        threshold = topk_values[:, -1].unsqueeze(1)
        mask = (sim_matrix >= threshold).float()

        return sim_matrix * mask

    def forward(self, source, target, alpha=0.9):
        # 拼接特征 [num_source + num_target, num_features]
        features = torch.cat([source, target], dim=0)

        # 1. 动态图权重计算
        adj_matrix = self._compute_adaptive_weights(features)

        # 2. 图消息传播（均值聚合）
        degree = torch.sum(adj_matrix, dim=1, keepdim=True)
        norm_adj = adj_matrix / (degree + 1e-6)  # 归一化
        propagated_features = torch.mm(norm_adj, features)

        # 3. 残差连接保留原始信息
        output = alpha * propagated_features + (1 - alpha) * features

        # 拆分结果
        num_source = source.size(0)
        return output[:num_source], output[num_source:]



