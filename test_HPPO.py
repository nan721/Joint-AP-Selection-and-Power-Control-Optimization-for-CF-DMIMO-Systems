
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from Rician_environment import generate_position_matrix, get_distance_matrix

from statehppopow import State

from utils import create_directory, plot_bar_withMEN, plot_CDF_withMEN, plot_CDF_4_14
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser()
parser.add_argument('--mean', type=str, default='./compare_4.14/mean')
parser.add_argument('--CDF', type=str, default='./compare_4.14/CDF')
parser.add_argument('--position', type=str, default='./compare_4.14/position')
args = parser.parse_args()

GET_NEW_DATA = 1
CAPACITY_LOOP = 1
INDEX_NUMBER = 3000
GENERATE_NEW_POSITION = 1
user = 3
rrh = 10
EP_STEPS = State.SERVICE_NUMBER * user
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
def layer_init(layer, weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        if args1.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif args1.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        elif args1.weights_init == "kaiming":
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if args1.bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)

class Actorh(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actorh, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)

        self.pi_d = nn.Linear(256, action_dim)
        self.mean = nn.Linear(256, 1)
        self.logstd = nn.Linear(256, 1)


    def forward(self, state):
        n = torch.relu(self.l1(state))

        return n


    def pi(self, state, softmax_dim=0):
        n = self.forward(state)
        prob = F.softmax(self.pi_d(n), dim=softmax_dim)
        mean = torch.sigmoid(self.mean(n))
        log_std = self.logstd(n)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        log_std = log_std.clamp(-4, 15)
        return prob, mean, log_std


    def save_checkpoint(self, checkpoint_file):
            torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中


def main():
    create_directory(args.position, sub_dirs=[''])
    create_directory(args.mean, sub_dirs=[''])
    create_directory(args.CDF, sub_dirs=[''])

    PPO_memory1 = method_memory()
    PPO_memory34 = method_memory()
    PPO_memory964 = method_memory()


    if GET_NEW_DATA:
        RRH_matrix = np.loadtxt('RRH_matrix.txt').reshape(rrh, 2)
        if GENERATE_NEW_POSITION:
            for i in range(INDEX_NUMBER):
                USER_matrix = generate_position_matrix(user, rrh)[1]
                np.savetxt('./compare_4.14/position/USER_matrix_{}.txt'.format(i), USER_matrix.view(float))



        NetworkPPOth1 = Actorh(state_dim=73, action_dim=30, net_width=256)
        NetworkPPOth1.load_state_dict(
            torch.load('./network/hppo9-4-4_network_60000.pth', map_location=torch.device(device)))

        NetworkPPOth34 = Actorh(state_dim=73, action_dim=30, net_width=256)
        NetworkPPOth34.load_state_dict(
            torch.load('./network/hppo9-5-4_network_60000.pth', map_location=torch.device(device)))

        NetworkPPOth964 = Actorh(state_dim=73, action_dim=30, net_width=256)
        NetworkPPOth964.load_state_dict(
            torch.load('./network/hppo9-6-4_network_60000.pth', map_location=torch.device(device)))


        for i in range(INDEX_NUMBER):
            distance_matrix = get_distance_matrix(np.loadtxt('./compare_4.14/position/USER_matrix_{}.txt'.format(i)),
                                                  RRH_matrix)
            print('正在计算第{}轮分布'.format(i))


            T3 = time.time()
            selection_pairs = np.zeros((user, rrh))
            a = []
            selection_index = np.zeros((1, 2))
            R = np.zeros((1, 3))
            action_c_single = 1
            power_matrix = np.ones((user, rrh))
            log_prob_c = np.ones((user, rrh))
            power_10 = np.zeros((1, rrh))
            s_hppo = State(selection_pairs, distance_matrix, selection_index, R, power_matrix, power_10,
                           action_c_single)

            for j in range(EP_STEPS):
                s_hppo.largescale_matrix = normsac(s_hppo.largescale_matrix).reshape(1, user * rrh)

                s_hppo.action_c_single = torch.tensor(s_hppo.action_c_single, dtype=torch.float64)
                inputsacth2 = np.concatenate(
                    (s_hppo.selection_pairs.reshape(1, 30), s_hppo.power_10, s_hppo.R.reshape(1, user),
                     s_hppo.largescale_matrix), axis=1)

                inputsacth2 = torch.from_numpy(inputsacth2).type(torch.FloatTensor).to(device)

                pi, mean, log_std = NetworkPPOth1.pi(inputsacth2, softmax_dim=1)

                action_d = torch.argmax(pi).item()
                action_c_single = mean.item()
                s_hppo.selection_index, action_c_single = s_hppo.excute_static_action_with_service_number(action_d,
                                                                                                          action_c_single)

                s_hppo.R = s_hppo.get_expect_capacity(CAPACITY_LOOP)
            capacitySACth2 = s_hppo.get_expect_capacity(CAPACITY_LOOP).sum()
            PPO_memory1.update_capacity(capacitySACth2, i)
            print('hppo', capacitySACth2)


            selection_pairs = np.zeros((user, rrh))
            a = []
            selection_index = np.zeros((1, 2))
            R = np.zeros((1, 3))
            action_c_single = 1
            power_matrix = np.ones((user, rrh))
            log_prob_c = np.ones((user, rrh))
            power_10 = np.zeros((1, rrh))
            s_hppo = State(selection_pairs, distance_matrix, selection_index, R, power_matrix,power_10, action_c_single)

            for j in range(EP_STEPS):
                s_hppo.largescale_matrix = normsac(s_hppo.largescale_matrix).reshape(1, user * rrh)

                s_hppo.action_c_single = torch.tensor(s_hppo.action_c_single, dtype=torch.float64)
                inputsacth2 = np.concatenate(
                    (s_hppo.selection_pairs.reshape(1, 30), s_hppo.power_10, s_hppo.R.reshape(1, user), s_hppo.largescale_matrix), axis=1)

                inputsacth2 = torch.from_numpy(inputsacth2).type(torch.FloatTensor).to(device)

                pi, mean, log_std = NetworkPPOth34.pi(inputsacth2, softmax_dim=1)

                action_d = torch.argmax(pi).item()
                action_c_single = mean.item()
                s_hppo.selection_index, action_c_single = s_hppo.excute_static_action_with_service_number(action_d,
                                                                                                          action_c_single)

                s_hppo.R = s_hppo.get_expect_capacity(CAPACITY_LOOP)
            capacitySACth2 = s_hppo.get_expect_capacity(CAPACITY_LOOP).sum()
            PPO_memory34.update_capacity(capacitySACth2, i)
            print('hppo', capacitySACth2)




            selection_pairs = np.zeros((user, rrh))
            a = []
            selection_index = np.zeros((1, 2))
            R = np.zeros((1, 3))
            action_c_single = 1
            power_matrix = np.ones((user, rrh))
            log_prob_c = np.ones((user, rrh))
            power_10 = np.zeros((1, rrh))
            s_hppo = State(selection_pairs, distance_matrix, selection_index, R, power_matrix,power_10, action_c_single)

            for j in range(EP_STEPS):
                s_hppo.largescale_matrix = normsac(s_hppo.largescale_matrix).reshape(1, user * rrh)

                s_hppo.action_c_single = torch.tensor(s_hppo.action_c_single, dtype=torch.float64)
                inputsacth2 = np.concatenate(
                    (s_hppo.selection_pairs.reshape(1, 30), s_hppo.power_10, s_hppo.R.reshape(1, user), s_hppo.largescale_matrix), axis=1)

                inputsacth2 = torch.from_numpy(inputsacth2).type(torch.FloatTensor).to(device)

                pi, mean, log_std = NetworkPPOth964.pi(inputsacth2, softmax_dim=1)

                action_d = torch.argmax(pi).item()
                action_c_single = mean.item()
                s_hppo.selection_index, action_c_single = s_hppo.excute_static_action_with_service_number(action_d,
                                                                                                          action_c_single)

                s_hppo.R = s_hppo.get_expect_capacity(CAPACITY_LOOP)
            capacitySACth2 = s_hppo.get_expect_capacity(CAPACITY_LOOP).sum()
            PPO_memory964.update_capacity(capacitySACth2, i)
            print('hppo', capacitySACth2)


        create_directory('./compare_4.14_data', sub_dirs=[''])



        np.save('compare_4.14_data/hPPO9-4-4_memory34.npy.npy', PPO_memory1.capacity)
        np.save('compare_4.14_data/hPPO9-5-4_memory34.npy', PPO_memory34.capacity)
        np.save('compare_4.14_data/hPPO9-6-4_memory34.npy', PPO_memory964.capacity)




        plot_CDF_4_14(PPO_memory1.capacity[:i + 1],PPO_memory34.capacity[:i + 1],PPO_memory964.capacity[:i + 1],PPO_memory64.capacity[:i + 1],
                      'Cumulative Distribution Function', args.CDF)



if __name__ == '__main__':
    T1 = time.time()
    main()
    T2 = time.time()
    print('程序总运行时间:%s秒' % (T2 - T1))
    print('666')
