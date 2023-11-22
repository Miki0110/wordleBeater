import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import os
from machine_learning.replayBuffer import ReplayBuffer
from machine_learning.model import DQN


class DQAgent:
    """
    DQN Agent that interacts with and learns from the environment.
    """
    def __init__(self, state_size, action_size, replay_memory_size=1e5, batch_size=64, gamma=0.95, learning_rate=1e-3,
                 target_tau=2e-3, update_rate=4, file_name="wordle_dqn_model.pth", training=True, seed=None):
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
        self.file_name = file_name

        # Model
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)

        if training:
            # Load the model if it exists
            if os.path.exists(file_name):
                # Ask the user if they want to load the model
                answer = input("A saved model was found. Do you want to load it? (y/n): ")
                if answer.lower() == 'y':
                    model_state = torch.load(file_name)
                    self.model.load_state_dict(model_state['model_state'])
                    self.target_model.load_state_dict(model_state['model_state'])
                    print("Model loaded.")
                else:
                    print("Model not loaded.")
        else:
            # Load the model if it exists
            if os.path.exists(file_name):
                # Load the model
                model_state = torch.load(file_name)
                self.model.load_state_dict(model_state['model_state'])
                self.target_model.load_state_dict(model_state['model_state'])
                print("Model loaded.")
            else:
                raise Exception("No model found.")

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

    def action_values(self, state):
        """
        Returns the action values for given state.
        Params
        ======
            state (array_like): current state
            returns (array_like): action values
        ======
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
        probabilities = torch.nn.functional.softmax(action_values, dim=1).cpu().data.numpy()
        return probabilities

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

        # Check for errors in training
        if torch.isnan(loss):
            raise Exception("NaN loss")

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

    def save_model(self):
        torch.save({'input_size': self.state_size,
                    'output_size': self.action_size,
                    'model_state': self.model.state_dict()},
                    self.file_name)
