import gym
import random
import numpy as np
import torch
import torch.nn as nn
from agent import Agent
env = gym.make("CartPole-v1", render_mode='human')
s, _ = env.reset()

n_episode = 5000
n_time_step = 200

n_state = len(s)
n_action = env.action_space.n

EPSILON_DECAY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FREQUENCY = 10

REWARD_BUFFER = np.empty(shape=n_episode)

agent = Agent(n_input=n_state, n_output=n_action)
for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_time_step):
        epsilon = np.interp(episode_i * n_time_step + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        random_sample = random.random()     # 采样

        if random_sample <= epsilon:
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s)

        s_next, r, done, _, info = env.step(a)     # 下一状态，reward，是否结束，其他信息
        agent.memo.add_memo(s, a, r, done, s_next)
        s = s_next  # 状态转移
        episode_reward += r     # 累计奖励

        if done:    # 游戏结束
            s, _ = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break

        if np.mean(REWARD_BUFFER[:episode_i]) >= 50:
            a = agent.online_net.act(s)
            s, r, done, _, info = env.step(a)
            env.render()

            if done:
                env.reset()

        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()   # 状态，action, reward, done or not, next state

        # 计算target
        target_q_values = agent.target_net(batch_s_)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = batch_r + agent.GAMMA * (1-batch_done) * max_target_q_values

        # 计算q_values
        q_values = agent.online_net(batch_s)
        # 神经网络输入状态s，神经网络输出q(s,a1),q(s,a2).....q(s,an)，q是q值,s是状态，a是相应的动作
        # 如果使用的是一个族batch，里面有很多s，则每一个s对应一组q值
        # 下面这句就是要收集所有的q_value，给每个s状态找到最大的q值，并收集起来
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

        # 计算损失
        loss = nn.functional.smooth_l1_loss(targets, a_q_values)

        # gradient descent
        agent.optimizer.zero_grad()
        loss.backward()     # 反向传播
        agent.optimizer.step()

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())  # 更新网络

        # 展示训练过程
        print("Episode:{}".format(episode_i))
        print("Avg.Reward:{}".format(np.mean(REWARD_BUFFER[:episode_i])))   # 从开始到现在的奖励平均值

