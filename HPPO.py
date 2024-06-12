from torch.distributions.normal import Normal
from torch.nn.utils import clip_grad_norm_
import copy
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_STD_MAX = 0.0
LOG_STD_MIN = -5.0
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

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
        return prob,mean, log_std


    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中


class Critic(nn.Module):
    def __init__(self, state_dim, net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, 256)

        self.C2 = nn.Linear(256, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))

        v = self.C2(v)
        return v
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)####保存模型等相关参数

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))####将预训练的参数权重加载到新的模型之中


class PPO_hybrid(object):
    def __init__(
            self,
            state_dim,net_width,
            action_dim,device,checkpoint_dir,
            gamma=0.99,
            lambd=0.95,

            lr=1e-4,
            clip_rate=0.2,
            K_epochs=10,
            batch_size=64,
            l2_reg=1e-3,
            entropy_coef=1e-3,
            adv_normalization=False,
            entropy_coef_decay=0.99,
    ):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.actor = Actor(state_dim, action_dim, net_width).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)

        self.critic = Critic(state_dim, net_width).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.s_dim = state_dim
        self.data = []

        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.optim_batch_size = batch_size
        self.l2_reg = l2_reg
        self.entropy_coef = entropy_coef
        self.adv_normalization = adv_normalization
        self.entropy_coef_decay = entropy_coef_decay

    def select_action(self, state):
        '''Stochastic Policy'''
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            #
            pi,mean, log_std = self.actor.pi(state, softmax_dim=1)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action_c = x_t.clamp(0,1)
            all_log_prob_c = normal.log_prob(x_t)
            all_log_prob_c -= torch.log(1.0 - action_c.pow(2) + 1e-8)
            log_prob_c = all_log_prob_c


            m = Categorical(pi)
            prob_d = m.probs
            action_d = m.sample().item()
            pi_a = pi[0][action_d].item()
            log_prob_d = torch.log(prob_d + 1e-8)

        return action_d, pi_a,action_c, log_prob_c

    def evaluate(self, state):
        '''Deterministic Policy'''
        with torch.no_grad():
            pi = self.actor.pi(state, softmax_dim=0)
            a = torch.argmax(pi).item()
        return a, 1.0

    def train(self,transition_dict):

        actions_combined = torch.tensor(transition_dict['actions'], dtype=torch.float).to(device)
        a = torch.tensor(actions_combined[:, :1], dtype=torch.int64)
        c_action = actions_combined[:, 1:]
        s = torch.tensor(transition_dict['states'],
                         dtype=torch.float).to(self.device)
         # 动作不再是float类型
        old_prob_a = torch.tensor(transition_dict['pi_a'], dtype=torch.float).view(-1, 1).to(
            self.device)
        old_prob_a_c = torch.tensor(transition_dict['log_prob_c'], dtype=torch.float).to(
            self.device)
        old_prob_a_c = old_prob_a_c.squeeze(dim=1)
        r = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        s_prime = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dw_mask = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        done_mask = dw_mask
        self.entropy_coef *= self.entropy_coef_decay  # exploring decay

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(device)
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # useful in some envs

        """PPO update"""
        # Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))

        for _ in range(self.K_epochs):
            # Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, s.shape[0]))

                '''actor update'''
                # 离散
                prob,mean, log_std = self.actor.pi(s[index], softmax_dim=1)
                entropy = Categorical(prob).entropy().view(-1,1)
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b))
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss =( -torch.min(surr1, surr2) - self.entropy_coef * entropy).mean()

                # 连续
                std = log_std.exp()
                continue_action_dist_now = Normal(mean, std)

                continue_action_dist_entropy = continue_action_dist_now.entropy().sum(1,keepdim=True)
                continue_action_logprob_now = continue_action_dist_now.log_prob(c_action[index])
                continue_ratios = torch.exp(
                    continue_action_logprob_now.sum(1, keepdims=True) - old_prob_a_c[index].sum(1,keepdims=True))
                continue_surr1 = continue_ratios * adv[index]
                continue_surr2 = torch.clamp(continue_ratios, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                continue_action_loss = (-torch.min(continue_surr1,
                                                   continue_surr2) - self.entropy_coef * continue_action_dist_entropy).mean()

                ac_loss = a_loss + continue_action_loss



                self.actor_optimizer.zero_grad()
                ac_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                '''critic update'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()
        return a_loss.mean(), c_loss, entropy
    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
