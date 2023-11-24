import numpy as np
import os
from wordleSim import WordleEnvironment
from model import read_word_list
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from agents import DQAgent


if __name__ == "__main__":

    # Initialize TensorBoard SummaryWriter
    os.makedirs('runs', exist_ok=True)
    # Check for past logs and increment the log number
    log_number = 0
    for file in os.listdir('runs'):
        if file.startswith('wordle_experiment'):
            log_number += 1
    writer = SummaryWriter(f'runs/wordle_experiment_{log_number}')


    file_name = "wordle_dqn_model.pth"
    # Initialize metrics
    total_steps = 0
    rolling_window_size = 100
    rolling_rewards = deque(maxlen=rolling_window_size)
    rolling_wins = deque(maxlen=rolling_window_size)

    # Hyperparameters
    epsilon = 1.0  # Starting exploration rate
    epsilon_min = 0.01  # Minimum exploration rate
    epsilon_decay = 0.999999  # Decay rate for exploration prob
    gamma = 0.8711  # Discount rate -> Value from hyperparameter optimization
    learning_rate = 2.55e-05  # Learning rate -> Value from hyperparameter optimization
    target_tau = 0.0208  # Soft update rate for target network -> Value from hyperparameter optimization

    batch_size = 128  # Number of experiences to sample from the replay buffer -> Value from hyperparameter optimization
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
    alphabet = 'abcdefghijklmnopqrstuvwxyzæøå'
    remaining_length = 1
    state_size = len(alphabet)*5 + remaining_length

    action_size = len(word_list)  # The number of possible actions (words)

    # Initialize the agent
    agent = DQAgent(state_size, action_size, gamma=gamma, batch_size=batch_size, learning_rate=learning_rate, target_tau=target_tau, file_name=file_name)

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