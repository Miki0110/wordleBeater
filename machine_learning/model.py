import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def read_word_list():
    with open('../wordle_list.txt', 'r', encoding='UTF-8', newline='\r\n') as f:
        words = [line.strip().lower() for line in f]
    return words

class DQN(nn.Module):
    def __init__(self, state_size, action_size, seed=None):
        super(DQN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 1024).to(self.device)  # 1024 neurons in the hidden layer
        self.fc2 = nn.Linear(1024, 1536).to(self.device)
        self.fc3 = nn.Linear(1536, action_size).to(self.device)  # Output is the number of possible actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
