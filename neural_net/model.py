import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from wordleSim import WordleEnvironment, convert_word_list
from collections import namedtuple, deque
import random


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(input_dim, 128, device=device)
        self.fc2 = nn.Linear(128, 128, device=device)
        self.fc3 = nn.Linear(128, output_dim, device=device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the target network to evaluation mode
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.99  # Discount factor

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(len(self.policy_net.fc3.weight))  # Random action
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(0)[1].item()  # Greedy action

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough experiences yet
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Convert lists of numpy arrays to single numpy arrays, then to tensors
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(np.array(batch.done), dtype=torch.uint8).to(self.device)

        # Compute Q-values and targets
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Compute loss and update policy network
        loss = nn.functional.smooth_l1_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def evaluate(agent, env, num_episodes):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(env.max_guesses):
            action = agent.select_action(state, 0)  # Greedy action selection
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break
        rewards.append(episode_reward)
    return sum(rewards) / num_episodes

if __name__ == '__main__':
    # Load the word list and create the environment
    with open('../wordle_list.txt', 'r', encoding='UTF-8', newline='\r\n') as f:
        word_list = [line.strip().lower() for line in f]
    env = WordleEnvironment(convert_word_list(word_list))

    # Hyperparameters
    input_dim = 912
    output_dim = len(word_list)
    # Set training parameters
    num_episodes = 1000
    batch_size = 64
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995

    agent = DQNAgent(input_dim, output_dim)
    with tqdm(range(num_episodes), desc="Training", unit="episode") as pbar:
        for episode in pbar:
            state = env.reset()  # Reset the environment for the new episode
            epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))  # Decay epsilon

            for t in range(env.max_guesses):  # Assume a maximum number of steps per episode
                action = agent.select_action(state, epsilon)  # Epsilon-greedy action selection
                next_state, reward, done = env.step(action)  # Take action in the environment

                # Store experience in replay buffer
                agent.memory.push(state, action, reward, next_state, done)

                # Update state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                agent.learn(batch_size)

                if done:
                    break  # Episode finished

            # Update the target network, copying all weights and biases in DQN
            if episode % 10 == 0:
                agent.update_target()

            # Evaluate the agent
            average_reward = evaluate(agent, env, 2000)
            pbar.set_description(f'Avg reward: {average_reward:.2f}')