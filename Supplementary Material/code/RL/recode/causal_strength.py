import numpy as np
import math
from scipy.special import psi


def adj_cs(adj_matrix, data):
    """
    计算每条因果边的因果强度
    :param adj_matrix: 因果邻接矩阵
    :param data: 观测数据
    :return: 因果强度矩阵
    """
    num = 0
    if type(data) != "numpy.ndarray":
        data = np.asarray(data)
    dim = adj_matrix.shape[0]
    strength_matrix = np.array(np.zeros((dim, dim)))
    for i in range(dim):
        for j in range(dim):
            if adj_matrix[i, j] == 1:
                strength_matrix[i, j] = causal_strength(data[:, i], data[:, j])
                num = num + 1
    print(num)
    return strength_matrix


def causal_strength(sam1, sam2, refMeasure=2):
    """
    计算两个变量间的因果强度
    :param sam1: 样本1
    :param sam2: 样本2
    :param refMeasure: 归一化方式
    :return: 因果强度 float
    """
    # 取实部
    sam1 = np.real(sam1)
    sam2 = np.real(sam2)
    # 检查输入的参数
    len1 = len(sam1)
    len2 = len(sam2)
    if len1 < 20:
        print("Not enough observations in sam1 (must be > 20)")
        exit(1)
    if len2 < 20:
        print("Not enough observations in sam2 (must be > 20)")
        exit(1)
    if len1 != len2:
        print("Lenghts of sam1 and sam2 must be equal")
        exit(1)
    # 归一化标准化处理
    if refMeasure == 1:  # 归一化
        sam1 = (sam1 - min(sam1)) / (max(sam1) - min(sam1))
        sam2 = (sam2 - min(sam2)) / (max(sam2) - min(sam2))
    if refMeasure == 2:   # 标准化
        sam1 = (sam1 - np.mean(sam1)) / np.std(sam1)
        sam2 = (sam2 - np.mean(sam2)) / np.std(sam2)
    if refMeasure != 1 and refMeasure != 2:
        print("Warning: unknown reference measure - no scaling applied")
        exit(1)
    # 熵估计
    ind1 = np.sort(sam1)
    ind2 = np.sort(sam2)

    # 样本1的熵估计
    hx = 0
    for i in range(len1 - 1):
        delta = ind1[i + 1] - ind1[i]
        if delta != 0:
            hx = hx + math.log(abs(delta))
    hx = hx / (len1 - 1) + psi(len1) - psi(1)

    # 样本2的熵估计
    hy = 0
    for i in range(len2 - 1):
        delta = ind2[i + 1] - ind2[i]
        if delta != 0:
            hy = hy + math.log(abs(delta))
    hy = hy / (len1 - 1) + psi(len2) - psi(1)

    # 计算因果强度
    strength = 1 / abs(hy - hx)

    return strength


