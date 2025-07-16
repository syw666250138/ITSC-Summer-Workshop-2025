import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys


# 检测是否在Jupyter环境中运行
def is_ipython():
    return 'ipykernel' in sys.modules


class CliffWalkingAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float = 0.1,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 0.9995,
            final_epsilon: float = 0.01,
            discount_factor: float = 0.95
    ):
        """初始化Q-Learning智能体

        参数:
            env: 训练环境
            learning_rate: Q值更新速率 (0-1)
            initial_epsilon: 初始探索率 (通常1.0)
            epsilon_decay: 每回合探索率衰减
            final_epsilon: 最小探索率 (通常0.01-0.1)
            discount_factor: 未来奖励折扣因子 (0-1)
        """
        self.env = env

        # Q表: 映射(状态, 动作)到期望奖励
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # 探索参数
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # 跟踪训练进度
        self.training_error = []
        self.episode_rewards = []
        self.episode_lengths = []

    def get_action(self, state: int) -> int:
        """使用epsilon-greedy策略选择动作

        返回:
            action: 0(上), 1(右), 2(下), 3(左)
        """
        # 探索: 随机动作
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # 利用: 选择最佳动作
        else:
            return np.argmax(self.q_table[state])

    def update(
            self,
            state: int,
            action: int,
            reward: float,
            next_state: int,
            terminated: bool
    ):
        """基于经验更新Q值 (Q-learning核心)"""
        # 下一状态的最佳Q值 (如果终止则为0)
        future_q_value = (not terminated) * np.max(self.q_table[next_state])

        # 目标Q值 (Bellman方程)
        target = reward + self.discount_factor * future_q_value

        # 当前估计误差
        temporal_difference = target - self.q_table[state, action]
        self.training_error.append(temporal_difference)

        # 更新Q值
        self.q_table[state, action] += self.lr * temporal_difference

    def decay_epsilon(self):
        """每回合后减少探索率"""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """获取当前策略 (每个状态的最佳动作)"""
        return np.argmax(self.q_table, axis=1)

    def visualize_policy(self):
        """可视化策略在网格上的表现"""
        policy = self.get_policy()
        grid = np.full((4, 12), ' ')

        # 标记悬崖
        grid[3, 1:11] = 'C'

        # 标记起点和终点
        grid[3, 0] = 'S'
        grid[3, 11] = 'G'

        # 添加动作箭头
        action_symbols = ['↑', '→', '↓', '←']
        for state in range(48):
            row = state // 12
            col = state % 12
            if grid[row, col] not in ['C', 'S', 'G']:
                grid[row, col] = action_symbols[policy[state]]

        # 打印网格
        print("\n当前策略:")
        for row in grid:
            print("|" + "|".join(row) + "|")

    def visualize_q_table(self, top_n=5):
        """可视化Q值最高的几个状态"""
        print("\nQ值最高的状态:")
        flat_indices = np.argsort(self.q_table.max(axis=1))[-top_n:]
        for idx in reversed(flat_indices):
            row = idx // 12
            col = idx % 12
            print(f"状态 {idx} (位置 [{row}, {col}]):")
            print(f"  上: {self.q_table[idx, 0]:.2f}")
            print(f"  右: {self.q_table[idx, 1]:.2f}")
            print(f"  下: {self.q_table[idx, 2]:.2f}")
            print(f"  左: {self.q_table[idx, 3]:.2f}")


# 训练参数
learning_rate = 0.1
n_episodes = 500
start_epsilon = 1.0
epsilon_decay = 0.9995  # 经过1000回合后衰减到约0.0067
final_epsilon = 0.01
discount_factor = 0.95

# 创建训练环境（无渲染模式）
env = gym.make("CliffWalking-v0", render_mode=None)
agent = CliffWalkingAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor
)

# 训练循环
for episode in tqdm(range(n_episodes), desc="训练进度"):
    state, info = env.reset()
    total_reward = 0
    step_count = 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        agent.update(state, action, reward, next_state, terminated)

        state = next_state
        total_reward += reward
        step_count += 1
        done = terminated or truncated

    # 记录回合结果
    agent.episode_rewards.append(total_reward)
    agent.episode_lengths.append(step_count)

    # 衰减探索率
    agent.decay_epsilon()

    # 每100回合显示一次策略 - 使用渲染环境展示
    if episode % 100 == 0:
        print(f"\n回合 {episode}: 总奖励 = {total_reward}, 步数 = {step_count}, ε = {agent.epsilon:.4f}")
        agent.visualize_policy()

        # 使用渲染环境展示当前策略
        print("在渲染环境中展示当前策略...")
        demo_env = gym.make("CliffWalking-v0", render_mode="human")
        demo_state, info = demo_env.reset()
        demo_done = False
        demo_steps = 0

        # 设置最大演示步数防止无限循环
        max_demo_steps = 50

        while not demo_done and demo_steps < max_demo_steps:
            demo_action = agent.get_action(demo_state)
            next_demo_state, demo_reward, terminated, truncated, info = demo_env.step(demo_action)

            # 在Jupyter中渲染需要特殊处理
            if is_ipython():
                from IPython import display

                display.clear_output(wait=True)
                plt.imshow(demo_env.render())
                plt.axis('off')
                plt.show()
            else:
                demo_env.render()

            time.sleep(0.2)  # 添加延迟以便观察

            demo_state = next_demo_state
            demo_steps += 1
            demo_done = terminated or truncated

        print(f"演示回合结束，步数: {demo_steps}")
        demo_env.close()
        time.sleep(0.5)  # 确保环境完全关闭

# 最终策略可视化
print("\n最终策略:")
agent.visualize_policy()
agent.visualize_q_table()

# 绘制训练结果
plt.figure(figsize=(15, 10))

# 奖励曲线
plt.subplot(2, 2, 1)
plt.plot(agent.episode_rewards)
plt.title("")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)

# 移动平均奖励
plt.subplot(2, 2, 2)
window = 50
rewards_ma = np.convolve(agent.episode_rewards, np.ones(window) / window, mode='valid')
plt.plot(rewards_ma)
plt.title(f"Episode Reward (window={window})")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid(True)

# 步数曲线
plt.subplot(2, 2, 3)
plt.plot(agent.episode_lengths)
plt.title("Step per Episode")
plt.xlabel("Episode")
plt.ylabel("Step")
plt.grid(True)

# 训练误差
plt.subplot(2, 2, 4)
plt.plot(agent.training_error)
plt.title("Training Error")
plt.xlabel("Train Step")
plt.ylabel("Error")
plt.grid(True)

plt.tight_layout()
plt.savefig("cliff_walking_training.png")
plt.show()



# 测试训练好的智能体
def test_agent(agent, env, num_episodes=10, render=False):
    """测试智能体性能 (无探索)"""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # 纯利用

    total_rewards = []
    total_steps = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        step_count = 0
        done = False

        while not done:
            if render:
                env.render()
                time.sleep(0.1)

            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            state = next_state
            episode_reward += reward
            step_count += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        total_steps.append(step_count)
        print(f"测试回合 {episode + 1}: 奖励 = {episode_reward}, 步数 = {step_count}")

    # 恢复原始探索率
    agent.epsilon = original_epsilon

    # 计算平均性能
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = np.mean(np.array(total_rewards) > -100)  # 避免掉崖

    print(f"\n测试结果 ({num_episodes}回合):")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_steps:.2f}")
    print(f"成功率: {success_rate:.1%}")
    print(f"最佳路径长度: {min(total_steps)}步")

    return avg_reward, avg_steps


# 测试智能体
print("\n开始测试...")
test_env = gym.make("CliffWalking-v0", render_mode="human")
test_agent(agent, test_env, num_episodes=10, render=True)
test_env.close()

# 打印最优策略
optimal_policy = agent.get_policy()
print("\n最优策略 (状态 → 动作):")
for i in range(0, 48, 12):
    print(f"状态 {i}-{i + 11}: {optimal_policy[i:i + 12]}")