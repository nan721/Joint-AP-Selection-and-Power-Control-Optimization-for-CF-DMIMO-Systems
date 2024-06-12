#
# 包含State类，表征强化学习的状态。该类包含三部分，其一为该场景下的AP分配情况，为user行rrh列二元矩阵；其二为该种分布情况下收发两端的大尺度衰落；其三为功率分配系数。
# 此外，该类也可以返回这种分布及AP分配情况下的全局平均容量，计算平均容量时会生成多次小尺度衰落，分布计算容量后取平均。
# 计算容量的函数由get_expect_capacity实现。
# 此外，State类还可处理动作的执行：当动作合法时，State类会改变它的AP分配情况矩阵为动作执行后的样子；当动作不合法时，State类不会发生变化。
# State类还可以调用graphic函数来进行直观的分布与选择情况显示。
# AP发射功率，每用户最多选择的AP数目在State类中定义。
# 本文件依赖于Rician_environment.py
# 运行该程序仅作为查错使用，main函数并无意义

# 2022.2.14
# 取消掉了mmWAve_environment环境，继续使用Rician_environment环境，但是要更改大尺度衰落

import math
import os
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from Rician_environment import get_channel_matrix, get_largescale_fading, get_precoder_matrix, generate_position_matrix


class State(object):
    yita = 10 ** (-60 / 10) / 1000  # 每个RRH的发射功率 -60 dBm
    N0 = 10 ** (-143 / 10) / 1000  # 噪声方差 -143 dBm
    SERVICE_NUMBER = 3

    def __init__(self, selection_pairs, distance_matrix, sele_index,R,power_matrix,power_10,action_c_single):
        self.selection_pairs = selection_pairs
        self.distance_matrix = distance_matrix
        self.largescale_matrix = get_largescale_fading(distance_matrix)
        self.user = self.selection_pairs.shape[0]
        self.rrh = self.selection_pairs.shape[1]
        self.Nt = 4
        self.Nr = 1
        # 以下几个变量是为了在贪婪算法中，固定小尺度衰落来使用的，在一轮中，小尺度衰落不发生变化
        self.channel_matrix = get_channel_matrix(self.distance_matrix, self.Nt, self.Nr)  ###大尺度衰落乘以小尺度衰落的值
        self.sele_index = sele_index
        self.R = R
        self.power_matrix = power_matrix
        self.power_10 = power_10
        self.action_c_single = action_c_single
    def curr_selection(self):
        return self.selection_pairs

    def curr_distance(self):
        return self.distance_matrix

    def curr_largescale_matrix(self):
        return self.largescale_matrix

    def new_yita_dbm(self, new_yita_dbm):######功率固定
        """用于更改分布式基站发射功率，dBm形式"""
        self.yita = 10 ** (new_yita_dbm / 10) / 1000

    def avaliable_static_action_with_service_number(self):
        """用于在静态动作，且含有service_number的情况下，判断是否可行
        返回含有可行的绝对动作的list"""
        avaliable = []
        t = 0
        ## 第i个UE 第j个AP
        #每行最多有SERVICE_NUMBER个1  每列只能有一个1 或者没有1
        ## 0 1 1 0 0 1 0 0 0 0
        ## 1 0 0 0 1 0 1 0 0 0
        ## 0 0 0 1 0 1 0 1 0 0  ##如果selection_pairs已经选好 则avaliable为[]
        for i in range(self.user): ## i:0-2
            for j in range(self.rrh):## j:0-9
                if 1 not in self.selection_pairs[:, j] and self.selection_pairs[i, :].sum() < self.SERVICE_NUMBER:
                    avaliable.append(t)
                t += 1
        return avaliable

        ## 0 0 1 0 0 1 0 0 0 0
        ## 0 0 0 0 1 0 1 0 0 0
        ## 0 0 0 1 0 1 0 1 0 0  ##如果selection_pairs没有选好 则avaliable为0 1 8 9 10 11 ...

    def excute_static_action_with_service_number(self, action,action_c_single):
        """适用于执行静态动作且有service_number的限制
        输入动作为绝对动作编号"""
        if action in self.avaliable_static_action_with_service_number():
            r = action // self.rrh
            c = action % self.rrh
            self.selection_pairs[r, c] = 1  ###如果动作可选，则置1
            self.power_matrix[r, c] = action_c_single
            self.action_c_single = action_c_single
            self.power_10[0, c] = action_c_single
        x = action // 10 + 1
        y = action % 10 + 1
        self.sele_index = np.array([[x, y]])
        return self.sele_index,self.action_c_single



    # def avaliable_dynamic_action_with_service_number(self):
    #     """用于在动态动作，且含有service_number的情况下，判断是否可行,返回一个0-1矩阵，可行动作位置会标号为1
    #     返回numpy矩阵形式"""
    #     avaliable = np.zeros((self.user, self.rrh))
    #     for i in range(self.user):
    #         for j in range(self.rrh):
    #             if 1 not in self.selection_pairs[:, j] and self.selection_pairs[i, :].sum() < self.SERVICE_NUMBER:
    #                 avaliable[i, j] = 1
    #     return avaliable
    #
    # def excute_dynamic_action_with_service_number(self, action):
    #     """
    #     用于执行动态动作
    #     :param action:执行第几个最近的动作。若输入0，则执行最近的，1则执行第二近的
    #     :return: 无
    #     """
    #     distance_matrix_reshape = self.distance_matrix.reshape(self.user * self.rrh)
    #     distance_matrix_reshape_sort = distance_matrix_reshape.argsort()
    #     distance_matrix_sort = distance_matrix_reshape_sort.reshape(self.user, self.rrh)
    #     i = deepcopy(action)
    #     while True:
    #         if self.avaliable_dynamic_action_with_service_number()[
    #             distance_matrix_sort[i // self.rrh, i % self.rrh] // self.rrh, distance_matrix_sort[
    #                                                                                i // self.rrh, i % self.rrh] % self.rrh] == 1:
    #             self.selection_pairs[
    #                 distance_matrix_sort[i // self.rrh, i % self.rrh] // self.rrh, distance_matrix_sort[
    #                     i // self.rrh, i % self.rrh] % self.rrh] = 1
    #             break
    #         else:
    #             i += 1
    #             if i // self.rrh >= self.user or i % self.rrh >= self.rrh:
    #                 # 经过测试，在训练过程中，可能会出现i过大的情况，此时设置为不执行任何操作
    #                 break

    def get_power(self, channel_matrix, precoder_matrix, user_index, type):
        """
        该函数可用于求解某个用户所受到的有用信号功率和干扰信号功率
        :param channel_matrix: list形式的信道
        :param precoder_matrix: list形式的预编码
        :param user_index: 求解哪个用户，就在此写哪个用户的编号
        :param type: 1为有用信号功率，0为干扰信号功率
        :return: 功率power
        """
        power = 0

        ## 0 1 1 0 0 1 0 0 0 0
        ## 1 0 0 0 1 0 1 0 0 0
        ## 0 0 0 1 0 1 0 1 0 0
        if type == 1:  ##求有用信号功率
            for j in range(self.rrh):
                if self.selection_pairs[user_index, j] == 1:## j:0-9
                    power += self.power_matrix[user_index, j]*abs(np.transpose(channel_matrix[user_index][j].conjugate()) @ precoder_matrix[user_index][
                        j] @ np.transpose(precoder_matrix[user_index][j].conjugate()) @ channel_matrix[user_index][j])  ##@是什么意思
            return power
        else:  ##求干扰信号功率
            for j in range(self.rrh):
                ####if user_index = 0
                if self.selection_pairs[user_index, j] != 1 and self.selection_pairs[:, j].any() == 1:
                    power += self.power_matrix[np.where(self.selection_pairs[:, j] == 1)[0][0]][j]*abs(np.transpose(channel_matrix[user_index][j].conjugate()) @      ####channel_matrix[user_index][j] list形式
                                 precoder_matrix[np.where(self.selection_pairs[:, j] == 1)[0][0]][j] @ np.transpose(    ####返回第几行
                        precoder_matrix[np.where(self.selection_pairs[:, j] == 1)[0][0]][j].conjugate()) @
                                 channel_matrix[user_index][j])  ####np.where(matrix == 1)[0]返回行中为1的索引
            return power

    def get_power_in_small_cell(self, channel_matrix, precoder_matrix, user_index, type):
        """
        该函数可用于求解一个基站被多个用户重复选择时，某个用户所受到的有用信号功率和干扰信号功率
        重复选择时，为每个选择它的用户平均分配功率
        :param channel_matrix: list形式的信道
        :param precoder_matrix: list形式的预编码
        :param user_index: 求解哪个用户，就在此写哪个用户的编号
        :param type: 1为有用信号功率，0为干扰信号功率
        :return: 功率power
        """
        ## 0 1 1 0 0 1 0 0 0 0
        ## 1 0 0 0 1 0 1 0 0 0
        ## 1 0 0 1 0 1 0 1 0 0
        power = 0
        if type == 1:  ##求有用信号功率
            for j in range(self.rrh): ## j:0-9
                if self.selection_pairs[user_index, j] == 1:
                    power += (1 / self.selection_pairs[:, j].sum()) * abs(
                        np.transpose(channel_matrix[user_index][j]) @ precoder_matrix[user_index][
                            j] @ np.transpose(precoder_matrix[user_index][j]) @ channel_matrix[user_index][j])
            return power
        else:  ##求干扰信号功率
            column_sum = self.selection_pairs.sum(axis=0)   ####axis=0表示按列相加
        ## 2 1 1 1 1 1 2 1 1 0
            column_sum[column_sum == 0] = 1
        ## 2 1 1 1 1 1 2 1 1 1
            selection_pairs = self.selection_pairs / column_sum
        ##  0  1 1 0 0 0.5 0 0 0 0
        ## 0.5 0 0 0 1  0  1 0 0 0
        ## 0.5 0 0 1 0 0.5 0 1 0 0
            selection_pairs[user_index, :] = 0
        ## if user_index = 0
        ##  0  0 0 0 0  0  0 0 0 0
        ## 0.5 0 0 0 1  0  1 0 0 0
        ## 0.5 0 0 1 0 0.5 0 1 0 0
            for j in range(self.rrh): ## j:0-9
                if selection_pairs[:, j].sum() != 0:
                    power += selection_pairs[:, j].sum() * abs(np.transpose(channel_matrix[user_index][j]) @
                                                               precoder_matrix[
                                                                   np.where(self.selection_pairs[:, j] == 1)[0][0]][
                                                                   j] @ np.transpose(
                        precoder_matrix[np.where(self.selection_pairs[:, j] == 1)[0][0]][j]) @ channel_matrix[user_index][j])
            return power

    def get_instant_capacity(self):
        """
        这个函数专门应用于Greedy_method中，来计算瞬时容量的。走完一轮后，需要重新生成self.channel_matrix
        :return:
        """
        R = np.zeros((self.user, 1))
        if (self.selection_pairs == np.zeros((self.user, self.rrh))).all():
            return R
        useful_power = np.zeros((self.user, 1))
        interference_power = np.zeros((self.user, 1))
        precoder_matrix = get_precoder_matrix(self.channel_matrix)
        for i in range(self.user):
            useful_power[i] = self.yita * self.get_power(self.channel_matrix, precoder_matrix, i, type=1)
            interference_power[i] = self.yita * self.get_power(self.channel_matrix, precoder_matrix, i, type=0)
            R[i] += math.log(1 + (useful_power[i] / (interference_power[i] + self.N0)), 2)
        return R

    def new_channel_matrix(self):
        """
        该函数用于更新信道矩阵，在贪婪算法中，每进行完一轮选择后使用
        :return:
        """
        self.channel_matrix = get_channel_matrix(self.distance_matrix, self.Nt, self.Nr)

    def get_expect_capacity(self, loop):
        """
        计算每个用户的平均容量，loop为循环次数
        :return: 返回每个用户的容量（如果要求总容量需要.sum()函数）
        """
        R = np.zeros((self.user, 1))
        if (self.selection_pairs == np.zeros((self.user, self.rrh))).all():
            return R

        ####small cell
        if self.selection_pairs.sum(axis=0).__contains__(2) or self.selection_pairs.sum(axis=0).__contains__(3):
            for k in range(loop):
                useful_power = np.zeros((self.user, 1))
                interference_power = np.zeros((self.user, 1))
                channel_matrix = get_channel_matrix(self.distance_matrix, self.Nt, self.Nr)
                precoder_matrix = get_precoder_matrix(channel_matrix)
                for i in range(self.user):
                    useful_power[i] = self.get_power_in_small_cell(channel_matrix, precoder_matrix, i, type=1)
                    interference_power[i] = self.get_power_in_small_cell(channel_matrix, precoder_matrix, i, type=0)
                    R[i] += math.log(1 + (useful_power[i] / (interference_power[i] + self.N0)), 2)
            R /= loop
            return R
        for k in range(loop):
            useful_power = np.zeros((self.user, 1))
            interference_power = np.zeros((self.user, 1))
            channel_matrix = get_channel_matrix(self.distance_matrix, self.Nt, self.Nr)
            precoder_matrix = get_precoder_matrix(channel_matrix)
            for i in range(self.user):
                useful_power[i] = self.yita * self.get_power(channel_matrix, precoder_matrix, i, type=1)
                interference_power[i] = self.yita * self.get_power(channel_matrix, precoder_matrix, i, type=0)
                R[i] += math.log(1 + (useful_power[i] / (interference_power[i] + self.N0)), 2)
        R /= loop
        self.R = R
        return R

    def get_extra_reward(self, s, action, R):
        """
        用于计算对没有服务RRH的用户分配第一个RRH时的额外奖励
        返回该用户分配第一个RRH时该用户的容量
        :param loop:
        :return:
        """
        reward = np.zeros(R.shape)
        a = s.selection_pairs.sum(axis=1)   ####s.和self.区别？
        b = self.selection_pairs.sum(axis=1)
        c = action // self.rrh
        if b[c] == 1 and a[c] == 0:
            reward = -R
            reward[c] = 0
            reward = reward + R   #####？？？？？
        return reward

    def reset(self):
        """
        重置选择矩阵
        :return:
        """
        self.selection_pairs = np.zeros((self.user, self.rrh))

    def done(self, EP_STEPS, DONE):
        """
        最终状态标识符
        :param EP_STEPS: 检查是否已经达到最大步数限制
        :param DONE: 外部输入结束标识
        :return: 如果是最终状态则返回1，否则返回1
        """
        done = 0
        if self.selection_pairs.sum().sum() >= EP_STEPS or self.avaliable_static_action_with_service_number() == [] or DONE:
            done = 1
        return done

    def graphic(self, RRH_matrix, USER_matrix, reward):
        """
        绘图函数，reward需要输入
        :param RRH_matrix:
        :param USER_matrix:
        :param reward:
        :return:
        """
        plt.ion()
        plt.cla()
        x_rrh = RRH_matrix[:, 0]
        y_rrh = RRH_matrix[:, 1]
        plt.scatter(x_rrh, y_rrh, marker='*', color='blue', s=40, label='RRH')
        x_user = USER_matrix[:, 0]
        y_user = USER_matrix[:, 1]
        plt.scatter(x_user, y_user, marker='o', color='red', s=40, label='User')
        plt.legend(loc='best')
        plt.xlabel('X(m)' + " Reward: {:.5f} bit/s/Hz".format(reward.sum()))
        plt.ylabel('Y(m)')
        for i in range(self.selection_pairs.shape[0]):
            for j in range(self.selection_pairs.shape[1]):
                if self.selection_pairs[i, j] == 1:
                    dot1 = USER_matrix[i, :]
                    dot2 = RRH_matrix[j, :]
                    plt.plot([dot1[0], dot2[0]], [dot1[1], dot2[1]], color='green')
        plt.pause(0.00001)
        plt.show()  ####图在哪里？？？？

    def save_graphic(self, RRH_matrix, USER_matrix, reward, figure_file, index):
        plt.cla()
        x_rrh = RRH_matrix[:, 0]
        y_rrh = RRH_matrix[:, 1]
        plt.scatter(x_rrh, y_rrh, marker='*', color='blue', s=40, label='RRH')
        x_user = USER_matrix[:, 0]
        y_user = USER_matrix[:, 1]
        plt.scatter(x_user, y_user, marker='o', color='red', s=40, label='User')
        plt.legend(loc='best')
        plt.xlabel('X(m)' + " Reward: {:.5f} bit/s/Hz".format(reward.sum()))
        plt.ylabel('Y(m)')
        for i in range(self.selection_pairs.shape[0]):
            for j in range(self.selection_pairs.shape[1]):
                if self.selection_pairs[i, j] == 1:
                    dot1 = USER_matrix[i, :]
                    dot2 = RRH_matrix[j, :]
                    plt.plot([dot1[0], dot2[0]], [dot1[1], dot2[1]], color='green')
        plt.savefig(os.path.join(figure_file, 'selection_image_{}'.format(index)))


if __name__ == '__main__':
    # 测试求解有用干扰功率函数
    user = 2
    rrh = 2
    # RRH_matrix = np.loadtxt('RRH_matrix.txt').reshape(rrh, 2)
    # USER_matrix = np.loadtxt('USER_matrix.txt').reshape(user, 2)
    # distance_matrix = np.loadtxt('distance_matrix.txt').reshape(user, rrh)
    RRH_matrix, USER_matrix, distance_matrix = generate_position_matrix(user, rrh)
    largescale_fading = get_largescale_fading(distance_matrix)
    # smallscale_fading = get_smallscale_fading(1)
    channel_matrix = get_channel_matrix(distance_matrix, 4, 1)
    precoder_matrix = get_precoder_matrix(channel_matrix)
    selection_pairs = np.zeros((user, rrh))
    selection_pairs[0, 0] = 1
    selection_pairs[1, 1] = 1
    s = State(selection_pairs, distance_matrix)
    print(abs(np.transpose(channel_matrix[0][0]) @ precoder_matrix[0][
        0] @ np.transpose(precoder_matrix[0][0]) @ channel_matrix[0][0]))
    print(s.get_power(channel_matrix, precoder_matrix, 0, 1))
    print(abs(np.transpose(channel_matrix[0][1]) @ precoder_matrix[1][
        1] @ np.transpose(precoder_matrix[1][1]) @ channel_matrix[0][1]))
    print(s.get_power(channel_matrix, precoder_matrix, 0, 0))
    print(s.get_expect_capacity(10))
    selection_pairs = np.zeros((user, rrh))
    selection_pairs[0, 1] = 1
    selection_pairs[1, 0] = 1
    s = State(selection_pairs, distance_matrix)
    print(abs(np.transpose(channel_matrix[0][1]) @ precoder_matrix[0][
        1] @ np.transpose(precoder_matrix[0][1]) @ channel_matrix[0][1]))
    print(s.get_power(channel_matrix, precoder_matrix, 0, 1))
    print(abs(np.transpose(channel_matrix[0][0]) @ precoder_matrix[1][
        0] @ np.transpose(precoder_matrix[1][0]) @ channel_matrix[0][0]))
    print(s.get_power(channel_matrix, precoder_matrix, 0, 0))
    print(s.get_expect_capacity(10))
    s.graphic(RRH_matrix, USER_matrix, s.get_expect_capacity(10).sum())
    print("666")


    # # 测试求解每用户容量函数
    # user = 1
    # rrh = 20
    # RRH_matrix = np.loadtxt('RRH_matrix.txt').reshape(rrh, 2)
    # USER_matrix = np.loadtxt('USER_matrix.txt').reshape(user, 2)
    # distance_matrix = np.loadtxt('distance_matrix.txt').reshape(user, rrh)
    # selection_pairs=np.zeros((user,rrh))
    # selection_pairs[0,0]=1
    # s = State(selection_pairs, distance_matrix)
    # print(s.get_expect_capacity(10))
