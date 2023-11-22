import optuna
from tqdm import tqdm
from agents import DQAgent
import numpy as np
from model import read_word_list
from wordleSim import WordleEnvironment
from torch.utils.tensorboard import SummaryWriter


def objective(trial, env, state_size, action_size, epsilon):
    writer = SummaryWriter(log_dir=f"runs/trial_{trial.number}")

    # Define the hyperparameters to be optimized
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.9999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    # Add other hyperparameters you want to optimize

    # Initialize the agent with these hyperparameters
    agent = DQAgent(state_size, action_size, gamma=gamma, batch_size=batch_size, learning_rate=learning_rate, file_name='optuna_model.pth')

    total_reward = 0
    num_episodes = 100000
    with tqdm(desc="Training Progress", total=num_episodes) as pbar:
        for episode in pbar:
            state = env.reset()[0].copy() / 3
            remaining_guesses_state = 1 / 6
            state = np.concatenate((state.flatten(), [remaining_guesses_state]))
            done = False
            episode_reward = 0

            while not done:
                action = agent.act(state, epsilon)
                feedback, reward, done = env.step(action)
                feedback_state = feedback[0].copy() / 3
                remaining_guesses_state = env.current_guess_index / 6
                next_state = np.concatenate((feedback_state.flatten(), [remaining_guesses_state]))
                agent.step(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                writer.add_scalar('Episode Reward', episode_reward, episode)

            total_reward += episode_reward
            average_reward = total_reward / (episode + 1)
            # Log the average reward to TensorBoard
            writer.add_scalar('Average Reward', average_reward, trial.number)
            # Update the progress bar
            pbar.write(f"Episode {episode}, Average Reward: {average_reward}")
            pbar.update(1)

        average_reward = total_reward / num_episodes
    # Close the writer once done
    writer.close()
    return average_reward


if __name__ == "__main__":
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
    state_size = len(alphabet) * 5 + remaining_length

    action_size = len(word_list)  # The number of possible actions (words)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, env, state_size, action_size, 0.01), n_trials=100)

    print('Best hyperparameters: {}'.format(study.best_trial.params))

    # After optimization, log the best trial information
    best_trial = study.best_trial
    print('Best trial: {}'.format(best_trial.params))

    writer = SummaryWriter(log_dir="runs/best_trial")
    writer.add_scalar('Best Average Reward', best_trial.value, 1)
    for key, value in best_trial.params.items():
        writer.add_text(f'best_params/{key}', str(value), 1)
    writer.close()