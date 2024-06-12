# 基本环境，可以生成一种分布情况，并计算分布式RRH与用户之间的莱斯衰落信道。
# 其中，大尺度与小尺度衰落信息可以分开计算，大尺度衰落系数定义为收发两端距离的-2次方，莱斯衰落信道的镜像路径能量与散射路径能将之比1:1。
# 本文件不依赖任何文件
# 注意：运行该程序可以重新生成用户位置与天线位置，但是在根目录下生成新用户位置后，强化学习的神经网络需要重新训练

import math
import random

import numpy as np


def distance(x1, y1, x2, y2):
    d = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return d


def get_distance_matrix(USER_matrix, RRH_matrix):
    user = USER_matrix.shape[0]
    rrh = RRH_matrix.shape[0]
    d = np.zeros((user, rrh))
    for i in range(user):
        for j in range(rrh):
            d[i, j] = distance(USER_matrix[i, 0], USER_matrix[i, 1], RRH_matrix[j, 0], RRH_matrix[j, 1])
    return d


def generate_position_matrix(user, rrh):
    RRH_matrix = np.random.rand(rrh, 2) * 20
    USER_matrix = np.random.rand(user, 2) * 20
    distance_matrix = get_distance_matrix(USER_matrix, RRH_matrix)
    return RRH_matrix, USER_matrix, distance_matrix


def get_largescale_fading(distance_matrix):
    # return distance_matrix ** (-2)    大尺度衰落为d的-4次方时使用
    # 参考文献：
    # Effective Channel Gain-Based Access Point Selection in Cell-Free Massive MIMO Systems
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9113300&tag=1
    # L_db = 10 * 1.7 * np.log10(distance_matrix) + 20 * np.log10(4 * np.pi / (3e8 / 28e9)) + np.random.normal(0, 1.2)
    L_db = 37.6 * np.log10(distance_matrix) + 35.3
    return (10 ** (-L_db / 10)) ** (0.5)


def get_smallscale_fading(k):
    # k为镜像路径能量与散射路径能将之比
    # random.seed(5)  # 固定小尺度衰落
    smallscale_fading = math.sqrt(k / (k + 1)) + math.sqrt(1 / (k + 1)) * (
            random.gauss(0, 1 / math.sqrt(2)) * 1 + random.gauss(0, 1 / math.sqrt(2)) * 1j)
    return smallscale_fading


def get_channel(x, y, Nt, Nr, largescale_fading):
    H = np.zeros((Nt, Nr)) + 1j * np.zeros((Nt, Nr))
    for i in range(Nt):
        H[i, :] = largescale_fading[x, y] * get_smallscale_fading(1)
    return H


def get_channel_matrix(distance_matrix, Nt, Nr):
    user = distance_matrix.shape[0]
    rrh = distance_matrix.shape[1]
    channel_matrix = []
    largescale_fading = get_largescale_fading(distance_matrix)
    for i in range(user):
        channel_matrix.append([])
        for j in range(rrh):
            h = get_channel(i, j, Nt, Nr, largescale_fading)
            channel_matrix[i].append(h)
    return channel_matrix ###大尺度衰落乘以小尺度衰落的值 list形式 g=beta*h


def get_precoder_matrix(channel_matrix):
    precoder_matrix = []
    for i in range(len(channel_matrix)):
        precoder_matrix.append([])
        for j in range(len(channel_matrix[0])):
            p = channel_matrix[i][j] / np.linalg.norm(channel_matrix[i][j], ord=2)
            precoder_matrix[i].append(p)
    return precoder_matrix  ###list形式 w=g/|g|


if __name__ == '__main__':
    # 运行该程序可以生成RRH_matrix,USER_matrix并储存起来
    user = 3
    rrh = 10

    # RRH_matrix, USER_matrix, distance_matrix = generate_position_matrix(user, rrh)
    # np.savetxt('RRH_matrix.txt', RRH_matrix.view(float))
    # np.savetxt('USER_matrix.txt', USER_matrix.view(float))
    # np.savetxt('distance_matrix.txt', distance_matrix.view(float))

    RRH_matrix = np.loadtxt('RRH_matrix.txt').reshape(rrh, 2)
    USER_matrix = np.loadtxt('USER_matrix.txt').reshape(user, 2)
    distance_matrix = np.loadtxt('distance_matrix.txt').reshape(user, rrh)
    largescale_fading = get_largescale_fading(distance_matrix)
    # smallscale_fading = get_smallscale_fading(1)
    # selection_pairs = np.zeros((user, rrh))

    channel_matrix = get_channel_matrix(distance_matrix, 4, 1)
    precoder_matrix = get_precoder_matrix(channel_matrix)
    print("666")
