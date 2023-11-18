import numpy as np
import os
import random
from wordleSim import WordleEnvironment
from model import DQN, read_word_list
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer:
    """
    Replay buffer for storing experiences and sampling batches for training.
    The replay buffer holds memories in the form of (state, action, reward, next_state, done).
    """
    def __init__(self, action_size, buffer_size, batch_size, seed=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=buffer_size)
        self.action_size = action_size
        self.batch_size = batch_size
        random.seed(seed)

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def append(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class DQAgent:
    """
    DQN Agent that interacts with and learns from the environment.
    """
    def __init__(self, state_size, action_size, replay_memory_size=1e5, batch_size=64, gamma=0.95, learning_rate=1e-3,
                 target_tau=2e-3, update_rate=4,  seed=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)

        # Model
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)

        # Load the model if it exists
        if os.path.exists('wordle_dqn_model.pth'):
            # Ask the user if they want to load the model
            answer = input("A saved model was found. Do you want to load it? (y/n): ")
            if answer.lower() == 'y':
                model_state = torch.load('wordle_dqn_model.pth')
                self.model.load_state_dict(model_state['model_state'])
                self.target_model.load_state_dict(model_state['model_state'])
                print("Model loaded.")
            else:
                print("Model not loaded.")

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        Params
        ======
            state (array_like): current state
            action (int): action index
            reward (float): reward
            next_state (array_like): next state
            done (bool): whether the episode is done or not
            returns (float): loss
        ======
        """
        # Save experience in replay memory
        self.memory.append(state, action, reward, next_state, done)
        loss = None
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                loss = self.learn(experiences, self.gamma)
        return loss

    def act(self, state, eps=0.0):
        """
        Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            returns (int): action index
        ======
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount rate
            returns (float): loss
        ======
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Qsa_prime_target_values = self.target_model(next_states).detach()
        Qsa_prime_target = Qsa_prime_target_values.max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Qsa_targets = rewards + (gamma * Qsa_prime_target * (1 - dones))

        # Compute loss
        loss = F.mse_loss(self.model(states).gather(1, actions), Qsa_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.model, self.target_model, self.tau)
        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, name='wordle_dqn_model.pth'):
        torch.save({'input_size': self.state_size,
                    'output_size': self.action_size,
                    'model_state': self.model.state_dict()},
                    name)


if __name__ == "__main__":

    # Initialize TensorBoard SummaryWriter
    os.makedirs('runs', exist_ok=True)
    # Check for past logs and increment the log number
    log_number = 0
    for file in os.listdir('runs'):
        if file.startswith('wordle_experiment'):
            log_number += 1
    writer = SummaryWriter(f'runs/wordle_experiment_{log_number}')

    # Initialize metrics
    total_steps = 0
    rolling_window_size = 100
    rolling_rewards = deque(maxlen=rolling_window_size)
    rolling_wins = deque(maxlen=rolling_window_size)

    # Hyperparameters
    epsilon = 1.0  # Starting exploration rate
    epsilon_min = 0.01  # Minimum exploration rate
    epsilon_decay = 0.999996  # Decay rate
    gamma = 0.95  # Discount rate
    learning_rate = 1e-2  # Learning rate

    batch_size = 64  # Number of experiences to sample from the replay buffer
    save_interval = 10000  # Save the model every n episodes

    # Set up the sim environment
    word_list = read_word_list()
    env = WordleEnvironment(word_list)

    """
    INPUT
    # [0-3]x30 5x encoded letters, 0 = not used, 1 = yellow, 2 = green, 3 = not in word
    # [1-6] 1x current guess
    
    OUTPUT
    [0-len(Wordlist)] 1x word index
    """

    # Calculate the state vector size
    alphabet = 'abcdefghijklmnopqrstuvwxyzæøåé'
    remaining_length = 1
    state_size = len(alphabet)*5 + remaining_length

    action_size = len(word_list)  # The number of possible actions (words)

    # Initialize the agent
    agent = DQAgent(state_size, action_size, gamma=gamma, batch_size=batch_size, learning_rate=learning_rate)

    # Main training loop
    try:
        episode = 0
        with tqdm(total=1, desc="Training Progress", unit=" episodes", position=0, ncols=80, leave=True, ascii=True) as pbar:
            with tqdm(total=1, desc="Episode Progress", unit=" steps", position=1, ncols=80, leave=False, ascii=True) as infobar:
                while True:
                    state = env.reset()[0].copy() / 3  # Reset the Wordle environment
                    remaining_guesses_state = 1 / 6

                    state = np.concatenate((state.flatten(), [remaining_guesses_state]))

                    done = False
                    total_reward = 0
                    steps = 0
                    episode += 1

                    while not done:
                        action = agent.act(state, epsilon)
                        feedback, reward, done = env.step(action)  # Perform the action in the environment
                        if np.array_equiv(feedback[1][-1], np.array([2, 2, 2, 2, 2])):  # Check if the game is won
                            rolling_wins.append(1)
                        elif done:
                            rolling_wins.append(0)
                        feedback_state = feedback[0].copy() / 3
                        remaining_guesses_state = env.current_guess_index / 6

                        next_state = np.concatenate((feedback_state.flatten(), [remaining_guesses_state]))

                        # Save the experience to the replay buffer
                        loss = agent.step(state, action, reward, next_state, done)
                        if loss is not None:
                            writer.add_scalar('Loss', loss, total_steps)

                        # Set new state
                        state = next_state

                        # Update metrics
                        total_reward += reward
                        total_steps += 1
                        steps += 1
                        infobar.update(1)

                    # Update rolling rewards
                    rolling_rewards.append(total_reward / steps)

                    # Log reward to TensorBoard
                    writer.add_scalar('Reward', np.mean(rolling_rewards), total_steps)

                    # Decay epsilon
                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay

                    # Save the model periodically
                    if episode % save_interval == 0:
                        agent.save_model()

                    # Update progress bars
                    if episode % rolling_window_size == 0:
                        # Update wins
                        pbar.set_description(f"Wins: {sum(rolling_wins)}/{len(rolling_wins)} = {np.mean(rolling_wins)*100:.2f}%")
                        infobar.set_description(
                            f"Avg Reward (Last {rolling_window_size} episodes): {np.mean(rolling_rewards):.4f}, Epsilon: {epsilon:.4f}")
                    pbar.update(1)
    except KeyboardInterrupt:
        print("Training stopped by user.")
        agent.save_model()

# Close the TensorBoard writer
writer.close()