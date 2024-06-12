# 训练函数，如要训练神经网络即直接运行此函数。AP位置不发生变动。
# 该函数可以直接调整HSAC的学习率、总学习轮数、记忆池大小、学习batch_size等参数。
# 最后一步不满足每个用户都有至少一个分布式基站时，给予负向rewawrd
# 天线位置不变，用户位置变化

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
from copy import deepcopy
import numpy as np
import argparse
from HSAC import SAC_hybrid
from Rician_environment import generate_position_matrix, get_distance_matrix
from state import State
from utils import plot_learning_curve, create_directory
import torch
def stepfun(x):  # 定义函数
    if x > 0:
        return np.array([0])  # 此时返回的是bool类型的数组
    else:
        return np.array([1])
parser = argparse.ArgumentParser()
# parser.add_argument('--max_episodes', type=int, default=200000)
parser.add_argument('--max_episodes', type=int, default=10000)
# parser.add_argument('--max_episodes', type=int, default=50000)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/HSAC/')
parser.add_argument('--reward_path', type=str, default='./output_images/avg_reward')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon')
parser.add_argument('--avg_rewardture_path', type=str, default='./output_images/avg_rewardture')
args = parser.parse_args()
def norm(largescale):
    largescale = (largescale - largescale.mean()) / largescale.std()
    return largescale
CHANGE_POSITION = 0
CHANGE_USER_POSITION = 1
# CHANGE_POSITION = 0
GRAPH_OUTPUT = 0
import buffer

def main():
    user = 3
    rrh = 10
    actor_lr = 1e-5
    critic_lr = 1e-4
    alpha_lr = 1e-4
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.9
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 100
    batch_size = 32
    target_entropy = -1
    state_dim=73
    out_d_dim=30
    out_c_dim = 1
    EP_STEPS = user * State.SERVICE_NUMBER
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    replay_buffer = buffer.ReplayBuffer(buffer_size)
    agent = SAC_hybrid(state_dim, out_c_dim, out_d_dim,  tau, gamma, device,args.ckpt_dir)

    
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    create_directory(args.reward_path, sub_dirs=[''])
    create_directory(args.epsilon_path, sub_dirs=[''])
    create_directory(args.avg_rewardture_path, sub_dirs=[''])
    rewards, avg_rewards, eps_history = [], [], []
    rewardtrue_list,rewardtrue_list,avg_rewardstrue = [], [], []
    # if not CHANGE_POSITION:
    #     RRH_matrix = np.loadtxt('RRH_matrix.txt').reshape(rrh, 2)
    #     USER_matrix = np.loadtxt('USER_matrix.txt').reshape(user, 2)
    #     distance_matrix = np.loadtxt('distance_matrix.txt').reshape(user, rrh)
    #     print("Position Matrix loaded")
    # RRH_matrix, USER_matrix, distance_matrix = generate_position_matrix(user, rrh)
    for episode in range(60000):
        # if CHANGE_POSITION:
        #     RRH_matrix, USER_matrix, distance_matrix = generate_position_matrix(user, rrh)
        if CHANGE_USER_POSITION:
            USER_matrix = generate_position_matrix(user, rrh)[1]
            if np.loadtxt('RRH_matrix.txt').shape[0] == rrh:
                RRH_matrix = np.loadtxt('RRH_matrix.txt').reshape(rrh, 2)
            else:
                RRH_matrix = generate_position_matrix(user, rrh)[0]
                np.savetxt('RRH_matrix.txt', RRH_matrix.view(float))
            distance_matrix = get_distance_matrix(USER_matrix, RRH_matrix)
        selection_pairs = np.zeros((user, rrh))

        a = []
        selection_index = np.zeros((1, 2))
        R = np.zeros((1, 3))
        action_c_single = 1
        power_matrix = np.zeros((user, rrh))
        power_10 = np.zeros((1, rrh))
        s = State(selection_pairs, distance_matrix, selection_index,R,power_matrix,power_10,action_c_single)
        for i in range(EP_STEPS):
            s.largescale_matrix = norm(s.largescale_matrix).reshape(1, user * rrh)
            observation = np.concatenate((s.selection_pairs.reshape(1,30),s.power_10.reshape(1,10), s.R.reshape(1, user),s.largescale_matrix), axis=1)

            USE  = deepcopy(observation)
            action_c_single,action_c, action_d = agent.take_action(USE)
            a.append(action_d)
            s_ = deepcopy(s)
            sele_index,action_c_single,power_10 = s_.excute_static_action_with_service_number(action_d,action_c_single)
            done = s.done(s.SERVICE_NUMBER * s.user, DONE=1 if i == EP_STEPS - 1 else 0)
            R = s_.get_expect_capacity(200)
            reward = R.sum()
            all_a = torch.cat([torch.from_numpy(np.array([[action_d]])), action_c.cpu()], dim=1)
            all_a = all_a.detach().numpy()
            observation_ = np.concatenate((s_.selection_pairs.reshape(1,30),s_.power_10.reshape(1,10), s_.R.reshape(1, user),s.largescale_matrix), axis=1)
            Rth=0.1
            s = deepcopy(s_)
            rewardtrue = 0
            tiaojian = []
            for index in range(user):
                tiaojian.append(stepfun(Rth - R[index][0]).tolist())
            if np.asarray(tiaojian).all() == 0:
                for index in range(user):
                    rewardtrue = rewardtrue - stepfun(R[index][0] - Rth)[0]
            else:
                rewardtrue = R.sum()

            replay_buffer.add(observation, all_a, rewardtrue, observation_, done)
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                b_s = np.squeeze(b_s)
                b_a = np.squeeze(b_a)
                b_ns = np.squeeze(b_ns)
                b_r = b_r.reshape(-1, 1)
                b_d = b_d.reshape(-1, 1)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict)

            if GRAPH_OUTPUT: s.graphic(RRH_matrix, USER_matrix, s.get_expect_capacity(loop=100).sum())
        print("Actions: ", a)
        print( s.selection_pairs)
        print( s.power_matrix)
        print(R.reshape(1,-1))
        rewards.append(R.sum())
        avg_reward = np.mean(rewards[-100:])
        avg_rewards.append(avg_reward)
        rewardtrue_list.append(rewardtrue)
        avg_rewardtrue = np.mean(rewardtrue_list[-100:])
        avg_rewardstrue.append(avg_rewardtrue)
        print('EP:{} Capacity:{} avg_reward:{} avg_rewardstrue:{}'.
              format(episode + 1, R.sum(), avg_reward, avg_rewardtrue))

        if (episode + 1) % 2000 == 0:
            agent.save_models(episode + 1)

    episodes = [i for i in range(episode + 1)]
    np.save('./avg_rewards94.npy', avg_rewards)
    np.save('./avg_rewardstrue94.npy', avg_rewardstrue)
    plot_learning_curve(episodes, avg_rewards, 'Final R', 'R.sum()', args.reward_path)
    plot_learning_curve(episodes, avg_rewardstrue, 'Final Reward', 'reward', args.avg_rewardture_path)
    s.graphic(RRH_matrix, USER_matrix, s.get_expect_capacity(loop=100).sum())


if __name__ == '__main__':
    T1 = time.time()
    main()
    T2 = time.time()
    print('程序运行时间:%s秒' % (T2 - T1))
    print('666')
