import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from wordleSim import WordleEnvironment

class RNNPolicy(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=4190):  # Output = len(word_list)
        super(RNNPolicy, self).__init__()
        # Set device to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Embedding layer to convert the input to a vector of size hidden_size
        self.embedding = nn.Embedding(26, 10).to(self.device)
        # Hidden size is the number of features in the hidden state h
        self.hidden_size = hidden_size
        # The RNN layer which takes an input sequence and the previous hidden state
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True).to(self.device)
        # A fully connected layer that outputs a vector of the size of the output space
        self.fc = nn.Linear(hidden_size, output_size).to(self.device)
        # Softmax layer to convert the output to a probability distribution
        self.softmax = nn.LogSoftmax(dim=2).to(self.device)

    def forward(self, x, hidden):
        # Split x into letter indices and feedback data
        letter_indices, feedback_data = x.split([5, input_size - 5 * 10], dim=-1)
        # Embed the letter indices
        letter_embeddings = self.embedding(letter_indices.long()).view(1, 1, -1)
        # Concatenate the letter embeddings and feedback data
        x = torch.cat([letter_embeddings, feedback_data], dim=-1)
        x = x.to(self.device)
        # x is (batch, seq_len, feature), hidden is (num_layers * num_directions, batch, hidden_size)
        x, hidden = self.rnn(x, hidden)
        # Forward pass through the fully connected layer
        x = self.fc(x)
        # Convert the output to a probability distribution
        x = self.softmax(x)
        return x, hidden

    def init_hidden(self, batch_size=1):
        # Initialize the hidden state to zeros
        # It should already be 3-D because we're initializing with batch dimension
        return torch.zeros(1, batch_size, self.hidden_size).to(self.device)


def train(model, environment, optimizer, num_episodes):
    epoc = 0
    while True:
        epoc += 1
        # Wrap the episode loop with tqdm for a progress bar
        pbar = tqdm(range(num_episodes), desc="Training", unit="episode")
        sum_reward = 0
        runs = 0
        for episode in pbar:
            correct_guesses = 0
            # Reset the environment and the RNN hidden state at the start of each episode
            state = environment.reset()
            hidden = model.init_hidden()
            log_probs = []
            rewards = []
            done = False

            while not done:
                # Prepare the state for input to the RNN
                letter_indices = torch.tensor(state['letter_indices'], dtype=torch.int64).unsqueeze(0)
                feedback_data = torch.tensor(state['feedback_data'], dtype=torch.float32).unsqueeze(0)
                # Convert the letter indices to one-hot encoding
                letter_one_hot = torch.nn.functional.one_hot(letter_indices, num_classes=29).float()
                # Flatten the one-hot encoding to match the expected input size
                letter_one_hot = letter_one_hot.view(1, 1, -1)
                state_tensor = torch.cat([letter_one_hot, feedback_data.unsqueeze(0).unsqueeze(0)], dim=-1)

                # Get the action probabilities and the next hidden state from the RNN
                action_probs, hidden = model(state_tensor, hidden)
                action_probs = action_probs.squeeze(0).squeeze(0)  # Remove batch and sequence dimensions

                # Sample an action from the action probabilities
                m = Categorical(action_probs.exp())  # exp() to convert log-probs to probs
                action = m.sample()

                # Take a step in the environment
                next_word = environment.word_list[action.item()]
                next_state, reward, done = environment.step(next_word)

                if reward == 10:
                    correct_guesses += 1

                # Store the log probability of the action taken and the reward received
                log_probs.append(m.log_prob(action))
                rewards.append(reward)

                # Update the state for the next iteration
                state = next_state

            # Compute the average reward per episode
            sum_reward += sum(rewards)
            runs += len(rewards)
            average_reward = sum_reward / runs
            # Compute the discounted return (assuming gamma=1 for simplicity)
            discounted_return = sum(rewards)

            # Compute the loss
            loss = -sum(log_prob * discounted_return for log_prob in log_probs)

            # Backpropagate the loss and update the model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the tqdm description with the latest action and reward
            pbar.set_description(f'Epoc {epoc}, Episode {episode + 1}, Average reward: {average_reward:.4f} Target: {env.target_word}, Correct: {correct_guesses}')
            pbar.refresh()  # Force update of description

        sum_reward = 0
        runs = 0

        if epoc % 50 == 0:
            # Save the model parameters to a file at the end of training
            torch.save(model.state_dict(), f'./models/model_epoc_{epoc}_params.pt')


if __name__ == '__main__':
    # Load the word list and create the environment
    with open('../wordle_list.txt', 'r', encoding='UTF-8', newline='\r\n') as f:
        word_list = [line.strip() for line in f]
    env = WordleEnvironment(word_list)

    # Create the RNN model
    input_size = 3*5 + 29*5  # 3 feedback states per letter, 29 letters, 5 letters
    rnn_policy = RNNPolicy(input_size)

    # Set up the optimizer
    optimizer = optim.Adam(rnn_policy.parameters(), lr=0.01)

    # Train the model
    num_episodes = 1000
    train(rnn_policy, env, optimizer, num_episodes)