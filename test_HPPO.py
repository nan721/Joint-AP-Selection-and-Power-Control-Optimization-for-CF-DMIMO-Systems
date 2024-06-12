
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
class Policypower(nn.Module):
    def __init__(self, input_shape, out_c, out_d):
        super(Policypower, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.mean = nn.Linear(256, out_c)
        self.logstd = nn.Linear(256, out_c)

        self.pi_d = nn.Linear(256, out_d)

        self.apply(layer_init)

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)

        x = torch.relu(self.fc1(x))
        mean = torch.tanh(self.mean(x))
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
        action_c = torch.sigmoid(x_t)
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

class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, out_c, out_d, layer_init):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape + out_c, 256)
        self.fc2 = nn.Linear(256, out_d)
        self.apply(layer_init)

    def forward(self, x, a, device):
        x = torch.Tensor(x).to(device)
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PolicyNetppo(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetppo, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        #print('x',x)
        x = F.relu(self.fc1(x))
        ##print('11111', x)
        return F.softmax(self.fc2(x), dim=1)
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中



class ValueNetppo(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetppo, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中





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

class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中
class PolicyNetdense3(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetdense3, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 256)
        self.fc2 = torch.nn.Linear(256, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.softmax(self.fc3(x), dim=1)

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中


class QValueNetdense3(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetdense3, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 256)
        self.fc2 = torch.nn.Linear(256, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
class PolicyNetdense4(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetdense4, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, action_dim)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=1)

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中


class QValueNetdense4(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetdense4, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 128)

        self.fc4 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中

class method_memory(nn.Module):
    def __init__(self):
        self.capacity = np.zeros((INDEX_NUMBER,))
        self.time = np.zeros((INDEX_NUMBER,))

    def update_capacity(self, capacity, i):
        self.capacity[i] = capacity

    def update_time(self, time, i):
        self.time[i] = time


def norm(state):
    state[0][state[0] == 0] = -1.0
    state[1] = (state[1] - state[1].mean()) / state[1].std()
    return state
def normsac(state):

    state = (state - state.mean()) / state.std()
    return state
class Actor1(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor1, self).__init__()

        self.l1 = nn.Linear(state_dim, 128)

        self.l2 = nn.Linear(128, action_dim)
    def forward(self, state):
        n = torch.relu(self.l1(state))

        return n
    def pi(self, state, softmax_dim=0):
        n = self.forward(state)
        prob = F.softmax(self.l2(n), dim=softmax_dim)
        return prob


    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中

class Actor2(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor2, self).__init__()

        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, action_dim)
    def forward(self, state):
        n = torch.relu(self.l1(state))
        n = torch.relu(self.l2(n))
        return n
    def pi(self, state, softmax_dim=0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob

class Actor3(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor3, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, action_dim)
    def forward(self, state):
        n = torch.relu(self.l1(state))
        n = torch.relu(self.l2(n))
        n = torch.relu(self.l3(n))
        return n
    def pi(self, state, softmax_dim=0):
        n = self.forward(state)
        prob = F.softmax(self.l4(n), dim=softmax_dim)
        return prob
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
class Actorh512(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actorh512, self).__init__()

        self.l1 = nn.Linear(state_dim, 512)

        self.pi_d = nn.Linear(512, action_dim)
        self.arfa = nn.Linear(512, 1)
        self.beta = nn.Linear(512, 1)
    def forward(self, state):
        n = torch.relu(self.l1(state))

        return n
    def pi(self, state, softmax_dim=0):
        n = self.forward(state)
        prob = F.softmax(self.pi_d(n), dim=softmax_dim)
        arfa = torch.relu(self.arfa(n))+1
        beta = torch.relu(self.beta(n))+1

        return prob,arfa, beta


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
            torch.load('./network/hppo9-4-4_q_eval_60000.pth', map_location=torch.device(device)))

        NetworkPPOth34 = Actorh(state_dim=73, action_dim=30, net_width=256)
        NetworkPPOth34.load_state_dict(
            torch.load('./network/hppo9-5-4_q_eval_60000.pth', map_location=torch.device(device)))

        NetworkPPOth964 = Actorh(state_dim=73, action_dim=30, net_width=256)
        NetworkPPOth964.load_state_dict(
            torch.load('./network/hppo9-6-4_q_eval_60000.pth', map_location=torch.device(device)))


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
