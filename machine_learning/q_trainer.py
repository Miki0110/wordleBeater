import numpy as np
import os
import random
from wordleSim import WordleEnvironment, convert_word_list
from model import DQN, choose_action, encode_guess, read_word_list, train_model
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import multiprocessing
from multiprocessing import Queue


class SharedReplayBuffer:
    def __init__(self, size):
        self.buffer = Queue(maxsize=size)

    def add(self, experience):
        if self.buffer.full():
            self.buffer.get()  # Remove oldest experience if the buffer is full
        self.buffer.put(experience)

    def sample(self, batch_size):
        samples = []
        for _ in range(batch_size):
            if self.buffer.empty():
                break
            samples.append(self.buffer.get())
        return samples

    def __len__(self):
        return self.buffer.qsize()


def worker(worker_id, env, model, replay_buffer, update_freq, queue, epsilon):
    step = 0
    state = env.reset()
    done = False

    while not done:
        # Select an action based on the current state and policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            action = model.predict(state)  # Exploitation using the current policy

        # Perform the action in the environment
        next_state, reward, done, _ = env.step(action)

        # Create an experience tuple
        experience = (state, action, reward, next_state, done)

        # Add experience to the shared replay buffer
        replay_buffer.add(experience)

        # Update state
        state = next_state if not done else env.reset()
        step += 1

        # Synchronize model periodically
        if worker_id == 0 and step % update_freq == 0:
            # Send local model state to the main process for synchronization
            queue.put(model.state_dict())

if __name__ == "__main__":
    # Experience replay buffer
    replay_buffer = deque(maxlen=10000)  # Adjust size as needed

    # Initialize metrics
    rolling_window_size = 100
    rolling_rewards = deque(maxlen=rolling_window_size)
    average_losses = deque(maxlen=rolling_window_size)
    shared_buffer = SharedReplayBuffer(size=10000)

    num_workers = 4  # Number of parallel workers
    update_freq = 100  # Frequency of model updates


    # Hyperparameters
    epsilon = 1.0  # Starting exploration rate
    epsilon_min = 0.01  # Minimum exploration rate
    epsilon_decay = 0.99999  # Decay rate
    gamma = 0.95  # Discount rate

    batch_size = 64  # Number of experiences to sample from the replay buffer
    save_interval = 1000  # Save the model every n episodes

    # Set up the sim environment
    word_list = read_word_list()
    env = WordleEnvironment(convert_word_list(word_list))

    # Calculate the state vector size
    alphabet = 'abcdefghijklmnopqrstuvwxyzæøåé'
    one_hot_length = len(alphabet)
    word_state_length = one_hot_length * 5
    feedback_state = 5
    #remaining_guesses_state = 1
    state_size = word_state_length + feedback_state #+ remaining_guesses_state

    action_size = len(word_list)  # The number of possible actions (words)

    # Set up the model
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()

    # Load the model if it exists
    if os.path.exists('wordle_dqn_model.pth'):
        # Ask the user if they want to load the model
        answer = input("A saved model was found. Do you want to load it? (y/n): ")
        if answer.lower() == 'y':
            model.load_state_dict(torch.load('wordle_dqn_model.pth'))
            print("Model loaded.")
        else:
            print("Model not loaded.")

    # Main training loop
    try:
        episode = 0
        with tqdm(total=1, desc="Training Progress", unit=" episodes", position=0, ncols=80, leave=True, ascii=True) as pbar:
            with tqdm(total=1, desc="Episode Progress", unit=" steps", position=1, ncols=80, leave=False, ascii=True) as infobar:
                while True:
                    state = env.reset()  # Reset the Wordle environment
                    no_word_vector = np.zeros(word_state_length)
                    state = np.concatenate((no_word_vector, state))

                    done = False
                    total_reward = 0
                    total_loss = 0
                    steps = 0
                    episode += 1

                    while not done:
                        action = choose_action(state, action_size, epsilon, model)
                        word_state = encode_guess(word_list[action])
                        feedback_state, reward, done = env.step(action)  # Perform the action in the environment

                        next_state = np.concatenate((word_state, feedback_state))
                        replay_buffer.append((state, action, reward, next_state, done))  # Store experience

                        total_reward += reward
                        steps += 1

                        # Check if the replay buffer is large enough for training
                        if len(replay_buffer) > batch_size:
                            batch = random.sample(replay_buffer, batch_size)
                            loss = train_model(optimizer, model, criterion, batch, gamma)  # Pass the entire batch for training
                            total_loss += loss.item()

                        state = next_state

                    # Update metrics
                    average_loss = total_loss / steps
                    average_losses.append(average_loss)
                    rolling_rewards.append(total_reward)

                    # Decay epsilon
                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay

                    # Save the model periodically
                    if episode % save_interval == 0:
                        torch.save(model.state_dict(), 'wordle_dqn_model.pth')

                    # Update progress bars
                    if episode % rolling_window_size == 0 and episode > 200:
                        infobar.set_description(
                            f"Avg Reward (Last {rolling_window_size} episodes): {np.mean(rolling_rewards):.4f}, Avg Loss: {np.mean(average_losses):.4f}, Epsilon: {epsilon:.4f}")
                    pbar.update(1)
                    infobar.update(1)
    except KeyboardInterrupt:
        print("Training stopped by user.")
        # Optionally save the model when training is stopped
        torch.save(model.state_dict(), 'wordle_dqn_model.pth')