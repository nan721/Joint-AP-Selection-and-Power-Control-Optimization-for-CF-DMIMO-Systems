from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ALGO LOGIC: initialize agent here:
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

args = parser.parse_args()
def layer_init(layer, weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        if args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        elif args.weights_init == "kaiming":
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if args.bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)

class Policy(nn.Module):
    def __init__(self, input_shape, out_c, out_d):
        super(Policy, self).__init__()
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


class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, out_c, out_d, layer_init):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape + out_c, 512)
        self.fc2 = nn.Linear(512, out_d)
        self.apply(layer_init)

    def forward(self, x, a, device):
        x = torch.Tensor(x).to(device)
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SAC_hybrid:
    ''' 处理混合动作的HSAC算法 '''
    def __init__(self, state_dim, out_c_dim, out_d_dim,  tau, gamma, device,checkpoint_dir):
        # 策略网络
        self.actor = Policy(state_dim, out_c_dim, out_d_dim).to(device)
        # 第一个Q网络
        self.critic_1 = SoftQNetwork(state_dim, out_c_dim, out_d_dim, layer_init).to(device)
        # 第二个Q网络
        self.critic_2 = SoftQNetwork(state_dim,  out_c_dim, out_d_dim, layer_init).to(device)
        self.target_critic_1 = SoftQNetwork(state_dim, out_c_dim, out_d_dim, layer_init).to(device)  # 第一个目标Q网络
        self.target_critic_2 = SoftQNetwork(state_dim, out_c_dim, out_d_dim, layer_init).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.values_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=1e-4)
        self.policy_optimizer = optim.Adam(list(self.actor.parameters()), lr=1e-3)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.starget_entropy_c = -0.99
        self.log_alpha_c = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_c = self.log_alpha_c.exp().detach().cpu().item()
        self.a_optimizer = optim.Adam([self.log_alpha_c], lr=1e-4)

        self.target_entropy_d = 0.1498
        self.log_alpha_d = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_d = self.log_alpha_d.exp().detach().cpu().item()
        self.a_d_optimizer = optim.Adam([self.log_alpha_d], lr=1e-4)

        # self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.out_d_dim = out_d_dim
        self.out_c_dim = out_c_dim
        self.user = 3
        self.rrh = 10
        self.checkpoint_dir = checkpoint_dir

    def take_action(self, state):

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action_c, action_d, log_prob_c, log_prob_d, prob_d = self.actor.get_action(state,device)
        action_d = action_d.item()
        c = action_d
        action_c_single = action_c[0].item()
        return action_c_single,action_c, action_d

    def norm(self, state):
        state[0][state[0] == 0] = -1.0
        state[1] = (state[1] - state[1].mean()) / state[1].std()
        return state


    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions_combined = torch.tensor(transition_dict['actions'], dtype=torch.float).to(device)
        actions_d = actions_combined[:, :1]
        actions_c = actions_combined[:, 1:]
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 更新两个Q网络
        with torch.no_grad():
            next_state_actions_c, next_state_actions_d, next_state_log_pi_c, next_state_log_pi_d, next_state_prob_d = self.actor.get_action(next_states, device)
            qf1_next_target = self.target_critic_1.forward(next_states, actions_c, device)
            qf2_next_target = self.target_critic_2.forward(next_states, actions_c, device)

            min_qf_next_target = next_state_prob_d * (torch.min(qf1_next_target, qf2_next_target) - self.alpha_c * next_state_prob_d * next_state_log_pi_c - self.alpha_d * next_state_log_pi_d)
            next_q_value = torch.Tensor(rewards).to(
                device) + (1 - torch.Tensor(dones).to(device)) * self.gamma * (min_qf_next_target.sum(1)).view(-1).reshape(32,1)
        qf1_a_values = self.critic_1.forward(states, actions_c, device).gather(1, actions_d.long().view(-1, 1).to(
            device)).squeeze().view(-1).reshape(32,1)
        qf2_a_values = self.critic_2.forward(states, actions_c, device).gather(1, actions_d.long().view(-1, 1).to(
            device)).squeeze().view(-1).reshape(32,1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = (qf1_loss + qf2_loss) / 2
        self.values_optimizer.zero_grad()
        qf_loss.backward()
        self.values_optimizer.step()

        # 更新策略网络
        actions_c, actions_d, log_pi_c, log_pi_d, prob_d = self.actor.get_action(states, device)
        qf1_pi = self.critic_1.forward(states, actions_c, device)
        qf2_pi = self.critic_2.forward(states, actions_c, device)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss_d = (prob_d * (self.alpha_d * log_pi_d - min_qf_pi)).sum(1).mean()
        policy_loss_c = (prob_d * (self.alpha_c * prob_d * log_pi_c - min_qf_pi)).sum(1).mean()
        policy_loss = policy_loss_d + policy_loss_c

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 更新alpha值
        with torch.no_grad():
            a_c, a_d, lpi_c, lpi_d, p_d = self.actor.get_action(states, device)
        alpha_loss = (-self.log_alpha_c * p_d * (p_d * lpi_c + self.starget_entropy_c)).sum(1).mean()
        alpha_d_loss = (-self.log_alpha_d * p_d * (lpi_d + self.target_entropy_d)).sum(1).mean()

        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha_c = self.log_alpha_c.exp().item()

        self.a_d_optimizer.zero_grad()
        alpha_d_loss.backward()
        self.a_d_optimizer.step()
        self.alpha_d = self.log_alpha_d.exp().item()


        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
