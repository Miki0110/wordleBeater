import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

def read_word_list():
    with open('../wordle_list.txt', 'r', encoding='UTF-8', newline='\r\n') as f:
        words = [line.strip().lower() for line in f]
    return words

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(state_size, 128).to(self.device)  # 128 neurons in the hidden layer
        self.relu = nn.ReLU().to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)
        self.fc2 = nn.Linear(128, 64).to(self.device)
        self.fc3 = nn.Linear(64, action_size).to(self.device)  # Output is the number of possible actions

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x


def train_model(optimizer, model, criterion, batch, gamma):
    states, actions, rewards, next_states, dones = zip(*batch)

    states_tensor = torch.FloatTensor(np.array(states)).to(model.device)
    next_states_tensor = torch.FloatTensor(np.array(next_states)).to(model.device)
    actions_tensor = torch.LongTensor(np.array(actions)).to(model.device)
    rewards_tensor = torch.FloatTensor(np.array(rewards)).to(model.device)
    dones_tensor = torch.FloatTensor(np.array(dones)).to(model.device)

    # Reshape tensors to match the dimensions expected by the neural network
    actions_tensor = actions_tensor.view(-1, 1)
    rewards_tensor = rewards_tensor.view(-1, 1)
    dones_tensor = dones_tensor.view(-1, 1)

    # Get current Q-value predictions
    current_q_values = model(states_tensor).gather(1, actions_tensor)

    # Calculate expected Q values
    with torch.no_grad():
        max_next_q_values = model(next_states_tensor).max(1)[0].view(-1, 1)
    expected_q_values = rewards_tensor + (gamma * max_next_q_values * (1 - dones_tensor))

    # Compute loss
    loss = criterion(current_q_values, expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def choose_action(state, action_size, epsilon, model):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_size - 1)
    else:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(model.device)
        with torch.no_grad():
            q_values = model(state_tensor)
        return torch.argmax(q_values).item()


def encode_guess(guess, alphabet='abcdefghijklmnopqrstuvwxyzæøåé'):
    """
    One-hot encodes a given guess.
    :param guess: The guessed word (e.g., 'apple')
    :param alphabet: The alphabet used (standard is 'abcdefghijklmnopqrstuvwxyzæøåé')
    :return: A list of encoded letters
    """
    encoded_guess = np.zeros(len(guess))
    for i, letter in enumerate(guess):
        index = alphabet.index(letter)
        encoded_guess[i] = index
    return encoded_guess


if __name__ == "__main__":
    alphabet = 'abcdefghijklmnopqrstuvwxyzæøåé'

    word_list = read_word_list()

    # Calculate the state vector size
    one_hot_length = len(alphabet)
    word_state = one_hot_length * 5
    feedback_state = 5
    remaining_guesses_state = 1
    state_size = word_state + feedback_state + remaining_guesses_state

    action_size = len(word_list)  # The number of possible actions (words)
    model = DQN(state_size, action_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
