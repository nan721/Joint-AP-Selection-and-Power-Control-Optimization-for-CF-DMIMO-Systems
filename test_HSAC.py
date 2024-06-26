# 重写比较，仅含有NN，MEN，RL,GM四种
import argparse
import time
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse

from Rician_environment import generate_position_matrix, get_distance_matrix

from state import State
from statepower import Statepower

from utils import create_directory, plot_bar_withMEN, plot_CDF_withMEN, plot_CDF_4_14

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser()
parser.add_argument('--mean', type=str, default='./compare_4.14/mean')
parser.add_argument('--CDF', type=str, default='./compare_4.14/CDF')
parser.add_argument('--position', type=str, default='./compare_4.14/position')
args = parser.parse_args()
create_directory(args.position, sub_dirs=[''])
create_directory(args.mean, sub_dirs=[''])
create_directory(args.CDF, sub_dirs=[''])
GET_NEW_DATA = 1
CAPACITY_LOOP = 1
INDEX_NUMBER = 3000
GENERATE_NEW_POSITION = 1
user = 3
rrh = 10
EP_STEPS = State.SERVICE_NUMBER * user
device = torch.device("cuda:0" if 0 else "cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中


class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DeepQNetwork(nn.Module):
    # Dueling DDQN
    def __init__(self, alpha, stack, action_dim):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(stack, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv3_value = nn.Conv2d(64, 4, kernel_size=1)
        self.conv3_adv = nn.Conv2d(64, 4, kernel_size=1)
        self.out_value = nn.Linear(4 * action_dim, 1)
        self.out_adv = nn.Linear(4 * action_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        value = F.relu(self.conv3_value(x))
        adv = F.relu(self.conv3_adv(x))
        value = value.view(value.size(0), -1)
        adv = adv.view(adv.size(0), -1)
        value = self.out_value(value)
        adv = self.out_adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage
        return Q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
LOG_STD_MAX = 0.0
LOG_STD_MIN = -5.0
parser = argparse.ArgumentParser(description='SAC with 2 Q functions, offline updates')
parser.add_argument('--weights-init',
                        default='kaiming',
                        const='kaiming',
                        nargs='?',
                        choices=['xavier', "orthogonal", 'uniform', 'kaiming'],
                        help='weight initialization scheme for the neural networks.')
parser.add_argument('--bias-init',
                        default='zeros',
                        const='xavier',
                        nargs='?',
                        choices=['zeros', 'uniform'],
                        help='weight initialization scheme for the neural networks.')

args1 = parser.parse_args()


class Policypower512(nn.Module):
    def __init__(self, input_shape, out_c, out_d):
        super(Policypower512, self).__init__()
        # self.embedding = nn.Embedding()
        # self.embeddings = nn.Embedding(64, 60)
        self.fc1 = nn.Linear(input_shape, 512)
        self.mean = nn.Linear(512, out_c)
        self.logstd = nn.Linear(512, out_c)

        self.pi_d = nn.Linear(512, out_d)

        self.apply(layer_init)

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = torch.relu(self.fc1(x))
        mean = torch.sigmoid(self.mean(x))
        log_std = self.logstd(x)
        pi_d = self.pi_d(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, pi_d
    def get_action(self, x, device):
        mean, log_std, pi_d = self.forward(x, device)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action_c = x_t.clamp(0,1)
        all_log_prob_c = normal.log_prob(x_t)
        all_log_prob_c -= torch.log(1.0 - action_c.pow(2) + 1e-8)
        log_prob_c = all_log_prob_c

        dist = Categorical(logits=pi_d)
        action_d = dist.sample()
        prob_d = dist.probs
        log_prob_d = torch.log(prob_d + 1e-8)

        return action_c, action_d, log_prob_c, log_prob_d, prob_d

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中


def main():
    SAC_memorySACth1 = method_memory()
    SAC_memorySACpowersc = method_memory()
    ex_memory = method_memory()
    excvx_memory = method_memory()
    exfp_memory = method_memory()
    SAC_memorySACpowerss = method_memory()
    SAC_memorySACpowert1s = method_memory()
    SAC_memorySACpowerts = method_memory()

    PPO_memory= method_memory()
    SAC_memory= method_memory()
    SAC_memorySACpowersc34 = method_memory()
    SAC_memoryhSAC256le5 = method_memory()
    SAC_memoryhSAC256le6 = method_memory()
    SAC_memoryhSAC256le7 = method_memory()
    SAC_memoryhSAC512le5 = method_memory()
    SAC_memoryhSAC512le6 = method_memory()
    SAC_memoryhSAC512le7 = method_memory()
    SAC_memoryhSAC1024le5 = method_memory()
    SAC_memoryhSAC1024le6 = method_memory()
    SAC_memoryhSAC1024le7 = method_memory()
 

    if GET_NEW_DATA:
        RRH_matrix = np.loadtxt('RRH_matrix.txt').reshape(rrh, 2)
        if GENERATE_NEW_POSITION:
            for i in range(INDEX_NUMBER):
                USER_matrix = generate_position_matrix(user, rrh)[1]
                np.savetxt('./compare_4.14/position/USER_matrix_{}.txt'.format(i), USER_matrix.view(float))




        NetworkhSAC256le5 = Policypower512(input_shape=73, out_c=1, out_d=30)
        NetworkhSAC256le5.load_state_dict(
            torch.load('./network/hsacle3_network_60000.pth', map_location=torch.device(device)))

        NetworkhSAC256le6 = Policypower512(input_shape=73, out_c=1, out_d=30)
        NetworkhSAC256le6.load_state_dict(
            torch.load('./network/hsacle4_network_60000.pth', map_location=torch.device(device)))

        NetworkhSAC256le7 = Policypower512(input_shape=73, out_c=1, out_d=30)
        NetworkhSAC256le7.load_state_dict(
            torch.load('./network/hsacle5_network_60000.pth', map_location=torch.device(device)))

        NetworkhSAC512le5 = Policypower512(input_shape=73, out_c=1, out_d=30)
        NetworkhSAC512le5.load_state_dict(
            torch.load('./network/hsacle6_network_60000.pth', map_location=torch.device(device)))

        NetworkhSAC512le6 = Policypower512(input_shape=73, out_c=1, out_d=30)
        NetworkhSAC512le6.load_state_dict(
            torch.load('./network/hsacle7_network_60000.pth', map_location=torch.device(device)))

        NetworkhSAC512le7 = Policypower512(input_shape=73, out_c=1, out_d=30)
        NetworkhSAC512le7.load_state_dict(
            torch.load('./network/hsac512-7_network_60000.pth', map_location=torch.device(device)))

       

        for i in range(INDEX_NUMBER):
            distance_matrix = get_distance_matrix(np.loadtxt('./compare_4.14/position/USER_matrix_{}.txt'.format(i)),
                                                  RRH_matrix)
            print('正在计算第{}轮分布'.format(i))



            sele_index = np.zeros((1, 2))
            R = np.zeros((1, 3))
            action_c_single = 1
            power_matrix = np.zeros((user, rrh))


            power_10 = np.zeros((1, rrh))
            hSAC256le5 = Statepower(np.zeros((user, rrh)), distance_matrix, sele_index, R, power_matrix,power_10,
                                             action_c_single)
            for j in range(EP_STEPS):
                hSAC256le5.largescale_matrix = normsac(hSAC256le5.largescale_matrix).reshape(1,user * rrh)
                inputhSAC256le5 = np.concatenate(
                    (hSAC256le5.selection_pairs.reshape(1, 30),hSAC256le5.power_10.reshape(1, 10),
                     hSAC256le5.R.reshape(1, user), hSAC256le5.largescale_matrix), axis=1)
                inputhSAC256le5 = inputhSAC256le5.astype(float)
                inputhSAC256le5 = torch.from_numpy(inputhSAC256le5).type(torch.FloatTensor).to(device)

                mean, log_std, pi_d = NetworkhSAC256le5(inputhSAC256le5, device)
                mean=mean.clamp(0,1)
                action_d = torch.argmax(pi_d).item()
                action_c_single = mean.item()
                hSAC256le5.excute_static_action_with_service_number(action_d, action_c_single)

                hSAC256le5.R = hSAC256le5.get_expect_capacity(CAPACITY_LOOP)
            capacityhSAC256le5 = hSAC256le5.get_expect_capacity(CAPACITY_LOOP).sum()
            SAC_memoryhSAC256le5.update_capacity(capacityhSAC256le5, i)
            print('hsac_sc', capacityhSAC256le5)
            print(hSAC256le5.power_matrix)


            sele_index = np.zeros((1, 2))
            R = np.zeros((1, 3))
            action_c_single = 1
            power_matrix = np.zeros((user, rrh))


            power_10 = np.zeros((1, rrh))
            hSAC256le6 = Statepower(np.zeros((user, rrh)), distance_matrix, sele_index, R, power_matrix,power_10,
                                             action_c_single)
            for j in range(EP_STEPS):
                hSAC256le6.largescale_matrix = normsac(hSAC256le6.largescale_matrix).reshape(1,user * rrh)
                inputhSAC256le6 = np.concatenate(
                    (hSAC256le6.selection_pairs.reshape(1, 30),hSAC256le6.power_10.reshape(1, 10),
                     hSAC256le6.R.reshape(1, user), hSAC256le6.largescale_matrix), axis=1)
                inputhSAC256le6 = inputhSAC256le6.astype(float)
                inputhSAC256le6 = torch.from_numpy(inputhSAC256le6).type(torch.FloatTensor).to(device)

                mean, log_std, pi_d = NetworkhSAC256le6(inputhSAC256le6, device)
                mean=mean.clamp(0,1)
                action_d = torch.argmax(pi_d).item()
                action_c_single = mean.item()
                hSAC256le6.excute_static_action_with_service_number(action_d, action_c_single)

                hSAC256le6.R = hSAC256le6.get_expect_capacity(CAPACITY_LOOP)
            capacityhSAC256le6 = hSAC256le6.get_expect_capacity(CAPACITY_LOOP).sum()
            SAC_memoryhSAC256le6.update_capacity(capacityhSAC256le6, i)
            print('hsac_sc', capacityhSAC256le6)
            print(hSAC256le6.power_matrix)

            sele_index = np.zeros((1, 2))
            R = np.zeros((1, 3))
            action_c_single = 1
            power_matrix = np.zeros((user, rrh))

            power_10 = np.zeros((1, rrh))
            hSAC256le7 = Statepower(np.zeros((user, rrh)), distance_matrix, sele_index, R, power_matrix, power_10,
                                    action_c_single)
            for j in range(EP_STEPS):
                hSAC256le7.largescale_matrix = normsac(hSAC256le7.largescale_matrix).reshape(1, user * rrh)
                inputhSAC256le7 = np.concatenate(
                    (hSAC256le7.selection_pairs.reshape(1, 30), hSAC256le7.power_10.reshape(1, 10),
                     hSAC256le7.R.reshape(1, user), hSAC256le7.largescale_matrix), axis=1)
                inputhSAC256le7 = inputhSAC256le7.astype(float)
                inputhSAC256le7 = torch.from_numpy(inputhSAC256le7).type(torch.FloatTensor).to(device)

                mean, log_std, pi_d = NetworkhSAC256le7(inputhSAC256le7, device)
                mean = mean.clamp(0, 1)
                action_d = torch.argmax(pi_d).item()
                action_c_single = mean.item()
                hSAC256le7.excute_static_action_with_service_number(action_d, action_c_single)

                hSAC256le7.R = hSAC256le7.get_expect_capacity(CAPACITY_LOOP)
            capacityhSAC256le7 = hSAC256le7.get_expect_capacity(CAPACITY_LOOP).sum()
            SAC_memoryhSAC256le7.update_capacity(capacityhSAC256le7, i)
            print('hsac_sc', capacityhSAC256le7)
            print(hSAC256le7.power_matrix)
            # GM_memory.capacity *= 20

            sele_index = np.zeros((1, 2))
            R = np.zeros((1, 3))
            action_c_single = 1
            power_matrix = np.zeros((user, rrh))

            power_10 = np.zeros((1, rrh))
            hSAC512le5 = Statepower(np.zeros((user, rrh)), distance_matrix, sele_index, R, power_matrix, power_10,
                                    action_c_single)
            for j in range(EP_STEPS):
                hSAC512le5.largescale_matrix = normsac(hSAC512le5.largescale_matrix).reshape(1, user * rrh)
                inputhSAC512le5 = np.concatenate(
                    (hSAC512le5.selection_pairs.reshape(1, 30), hSAC512le5.power_10.reshape(1, 10),
                     hSAC512le5.R.reshape(1, user), hSAC512le5.largescale_matrix), axis=1)
                inputhSAC512le5 = inputhSAC512le5.astype(float)
                inputhSAC512le5 = torch.from_numpy(inputhSAC512le5).type(torch.FloatTensor).to(device)

                mean, log_std, pi_d = NetworkhSAC512le5(inputhSAC512le5, device)
                mean = mean.clamp(0, 1)
                action_d = torch.argmax(pi_d).item()
                action_c_single = mean.item()
                hSAC512le5.excute_static_action_with_service_number(action_d, action_c_single)

                hSAC512le5.R = hSAC512le5.get_expect_capacity(CAPACITY_LOOP)
            capacityhSAC512le5 = hSAC512le5.get_expect_capacity(CAPACITY_LOOP).sum()
            SAC_memoryhSAC512le5.update_capacity(capacityhSAC512le5, i)
            print('hsac_sc', capacityhSAC512le5)
            print(hSAC512le5.power_matrix)


            sele_index = np.zeros((1, 2))
            R = np.zeros((1, 3))
            action_c_single = 1
            power_matrix = np.zeros((user, rrh))

            power_10 = np.zeros((1, rrh))
            hSAC512le6 = Statepower(np.zeros((user, rrh)), distance_matrix, sele_index, R, power_matrix, power_10,
                                    action_c_single)
            for j in range(EP_STEPS):
                hSAC512le6.largescale_matrix = normsac(hSAC512le6.largescale_matrix).reshape(1, user * rrh)
                inputhSAC512le6 = np.concatenate(
                    (hSAC512le6.selection_pairs.reshape(1, 30), hSAC512le6.power_10.reshape(1, 10),
                     hSAC512le6.R.reshape(1, user), hSAC512le6.largescale_matrix), axis=1)
                inputhSAC512le6 = inputhSAC512le6.astype(float)
                inputhSAC512le6 = torch.from_numpy(inputhSAC512le6).type(torch.FloatTensor).to(device)

                mean, log_std, pi_d = NetworkhSAC512le6(inputhSAC512le6, device)
                mean = mean.clamp(0, 1)
                action_d = torch.argmax(pi_d).item()
                action_c_single = mean.item()
                hSAC512le6.excute_static_action_with_service_number(action_d, action_c_single)

                hSAC512le6.R = hSAC512le6.get_expect_capacity(CAPACITY_LOOP)
            capacityhSAC512le6 = hSAC512le6.get_expect_capacity(CAPACITY_LOOP).sum()
            SAC_memoryhSAC512le6.update_capacity(capacityhSAC512le6, i)
            print('hsac_sc', capacityhSAC512le6)
            print(hSAC512le6.power_matrix)

            sele_index = np.zeros((1, 2))
            R = np.zeros((1, 3))
            action_c_single = 1
            power_matrix = np.zeros((user, rrh))

            power_10 = np.zeros((1, rrh))
            hSAC512le7 = Statepower(np.zeros((user, rrh)), distance_matrix, sele_index, R, power_matrix, power_10,
                                    action_c_single)
            for j in range(EP_STEPS):
                hSAC512le7.largescale_matrix = normsac(hSAC512le7.largescale_matrix).reshape(1, user * rrh)
                inputhSAC512le7 = np.concatenate(
                    (hSAC512le7.selection_pairs.reshape(1, 30), hSAC512le7.power_10.reshape(1, 10),
                     hSAC512le7.R.reshape(1, user), hSAC512le7.largescale_matrix), axis=1)
                inputhSAC512le7 = inputhSAC512le7.astype(float)
                inputhSAC512le7 = torch.from_numpy(inputhSAC512le7).type(torch.FloatTensor).to(device)

                mean, log_std, pi_d = NetworkhSAC512le7(inputhSAC512le7, device)
                mean = mean.clamp(0, 1)
                action_d = torch.argmax(pi_d).item()
                action_c_single = mean.item()
                hSAC512le7.excute_static_action_with_service_number(action_d, action_c_single)

                hSAC512le7.R = hSAC512le7.get_expect_capacity(CAPACITY_LOOP)
            capacityhSAC512le7 = hSAC512le7.get_expect_capacity(CAPACITY_LOOP).sum()
            SAC_memoryhSAC512le7.update_capacity(capacityhSAC512le7, i)
            print('hsac_sc', capacityhSAC512le7)
            print(hSAC512le7.power_matrix)

           

        create_directory('./compare_4.14_data', sub_dirs=[''])

   


        np.save('./compare_4.14_data/SAC_memoryhSAC256le5.npy', SAC_memoryhSAC256le5.capacity)
        np.save('./compare_4.14_data/SAC_memoryhSAC256le6.npy', SAC_memoryhSAC256le6.capacity)
        np.save('./compare_4.14_data/SAC_memoryhSAC256le7.npy', SAC_memoryhSAC256le7.capacity)
        np.save('./compare_4.14_data/SAC_memoryhSAC512le5.npy', SAC_memoryhSAC512le5.capacity)
        np.save('./compare_4.14_data/SAC_memoryhSAC512le6.npy', SAC_memoryhSAC512le6.capacity)
        np.save('./compare_4.14_data/SAC_memoryhSAC512le7.npy', SAC_memoryhSAC512le7.capacity)


        plot_CDF_4_14(SAC_memoryhSAC256le5.capacity[:i + 1],SAC_memoryhSAC256le6.capacity[:i + 1],SAC_memoryhSAC256le7.capacity[:i + 1],SAC_memoryhSAC512le5.capacity[:i + 1],SAC_memoryhSAC512le6.capacity[:i + 1],SAC_memoryhSAC512le7.capacity[:i + 1],SAC_memoryhSAC1024le5.capacity[:i + 1],SAC_memoryhSAC1024le6.capacity[:i + 1],SAC_memoryhSAC1024le7.capacity[:i + 1],
                      'Cumulative Distribution Function', args.CDF)
        print(666)
  


if __name__ == '__main__':
    T1 = time.time()
    main()
    T2 = time.time()
    print('程序总运行时间:%s秒' % (T2 - T1))
    print('666')
