import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from PIL import Image


# 定义经验回放缓冲区
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# 定义深度Q网络 - 修复维度问题
class DQN(nn.Module):
    def __init__(self, action_dim, frame_stack=4):
        super(DQN, self).__init__()
        self.frame_stack = frame_stack

        self.conv = nn.Sequential(
            nn.Conv2d(frame_stack * 3, 32, kernel_size=8, stride=4),  # 输入通道 = 帧数 * 3 (RGB)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        # 重新组织张量维度: [batch, H, W, C] -> [batch, C, H, W]
        # 其中 C = 帧数 * 3 (RGB通道)
        x = x.permute(0, 3, 1, 2)
        return self.fc(self.conv(x))


# 定义Car Racing Agent
class CarRacingAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float = 1e-4,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 0.99995,
            final_epsilon: float = 0.1,
            discount_factor: float = 0.99,
            buffer_size: int = 50000,
            batch_size: int = 32,
            target_update_freq: int = 1000,
            frame_stack: int = 4
    ):
        self.env = env
        self.action_dim = env.action_space.n
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.target_update_freq = target_update_freq
        self.frame_stack = frame_stack

        # 初始化Q网络和目标网络
        self.policy_net = DQN(self.action_dim, frame_stack)
        self.target_net = DQN(self.action_dim, frame_stack)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 状态缓冲区
        self.state_buffer = deque(maxlen=frame_stack)

        # 跟踪训练进度
        self.losses = []
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def preprocess(self, state):
        """预处理状态：裁剪、调整大小、归一化"""
        # 转换为PIL图像进行裁剪
        img = Image.fromarray(state)
        img = img.crop((0, 0, 96, 84))  # 裁剪掉仪表盘部分
        img = img.resize((84, 84))
        return np.array(img) / 255.0

    def get_state(self):
        """获取当前堆叠状态 - 修复维度问题"""
        # 堆叠帧: (84, 84, 3) * N -> (84, 84, 3*N)
        return np.concatenate(list(self.state_buffer), axis=-1)

    def reset(self, initial_state):
        """重置状态缓冲区"""
        self.state_buffer.clear()
        processed = self.preprocess(initial_state)
        for _ in range(self.frame_stack):
            self.state_buffer.append(processed)
        return self.get_state()

    def get_action(self, state: np.ndarray) -> int:
        """使用epsilon-greedy策略选择动作"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # 添加批次维度
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, state, action, reward, next_state, done):
        """存储经验并更新网络 - 修复维度问题"""
        # 存储经验
        self.replay_buffer.push(state, action, reward, next_state, done)

        # 如果缓冲区不足，则跳过更新
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从缓冲区采样
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 转换数据为张量 - 注意维度处理
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done)

        # 计算当前Q值
        current_q = self.policy_net(state_batch).gather(1, action_batch)

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + (1 - done_batch) * self.discount_factor * next_q

        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.losses.append(loss.item())

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # 更新目标网络
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """减少探索率"""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


# 训练参数设置
learning_rate = 1e-4
n_episodes = 10
start_epsilon = 1.0
epsilon_decay = 0.9
final_epsilon = 0.1
discount_factor = 0.99
batch_size = 32
buffer_size = 50000
target_update_freq = 1000
frame_stack = 4

# 创建环境和Agent
env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = CarRacingAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
    batch_size=batch_size,
    buffer_size=buffer_size,
    target_update_freq=target_update_freq,
    frame_stack=frame_stack
)

# 训练循环
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    state = agent.reset(obs)
    total_reward = 0
    step_count = 0

    while True:
        # 获取动作
        action = agent.get_action(state)

        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 预处理和更新状态
        processed_next_obs = agent.preprocess(next_obs)
        agent.state_buffer.append(processed_next_obs)
        next_state = agent.get_state()

        # 存储经验并更新网络
        agent.update(state, action, reward, next_state, terminated or truncated)

        # 更新状态和计数器
        state = next_state
        total_reward += reward
        step_count += 1

        # 检查是否结束
        if terminated or truncated:
            agent.episode_rewards.append(total_reward)
            agent.episode_lengths.append(step_count)
            break

    # 减少探索率
    agent.decay_epsilon()

    # 每100回合打印进度
    if episode % 100 == 0:
        avg_reward = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(
            agent.episode_rewards)
        print(f"Episode {episode}, Epsilon: {agent.epsilon:.4f}, Avg Reward: {avg_reward:.2f}")


# 绘制训练结果
def get_moving_avgs(arr, window):
    """计算移动平均以平滑数据"""
    return np.convolve(np.array(arr), np.ones(window), 'valid') / window


plt.figure(figsize=(15, 10))

# 奖励曲线
plt.subplot(3, 1, 1)
rewards_ma = get_moving_avgs(agent.episode_rewards, 20)
plt.plot(rewards_ma)
plt.title("Episode Rewards (Moving Avg)")
plt.ylabel("Reward")
plt.grid(True)

# 长度曲线
plt.subplot(3, 1, 2)
lengths_ma = get_moving_avgs(agent.episode_lengths, 20)
plt.plot(lengths_ma)
plt.title("Episode Lengths (Moving Avg)")
plt.ylabel("Steps")
plt.grid(True)

# 损失曲线
plt.subplot(3, 1, 3)
if agent.losses:
    # 只取最近10000个损失值
    recent_losses = agent.losses[-10000:]
    losses_ma = get_moving_avgs(recent_losses, 100)
    plt.plot(losses_ma)
    plt.title("Training Loss (Moving Avg)")
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.grid(True)

plt.tight_layout()
plt.savefig("car_racing_training.png")
plt.show()


# 测试训练好的Agent
def test_agent(agent, env, num_episodes=5, render=True):
    """测试Agent性能"""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.01  # 最小探索

    total_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        state = agent.reset(obs)
        total_reward = 0
        frames = []

        while True:
            if render:
                frames.append(env.render())

            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)

            processed_next_obs = agent.preprocess(next_obs)
            agent.state_buffer.append(processed_next_obs)
            state = agent.get_state()

            total_reward += reward

            if terminated or truncated:
                total_rewards.append(total_reward)
                print(f"Episode {episode + 1}: Reward = {total_reward:.1f}, Steps = {len(frames)}")
                break



    agent.epsilon = old_epsilon

    # 计算平均性能
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


# 创建测试环境
test_env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
test_agent(agent, test_env)

