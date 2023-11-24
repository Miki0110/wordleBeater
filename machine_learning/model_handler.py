import numpy as np
import torch
import os
from wordleSim import WordleEnvironment
from machine_learning.agents import DQAgent

class ModelHandler:
    def __init__(self, word_list, file_name="machine_learning/wordle_dqn_model.pth"):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzæøå'
        self.file_name = file_name
        self.word_list = word_list
        self.filtered_word_list = self.word_list.copy()
        self.env = WordleEnvironment(self.word_list)
        self.current_state = np.zeros((5, len(self.alphabet)))  # 5x30 state vector, where 5 is the number of letters and 30 is the alphabet
        self.current_guess_index = 1
        self.agent = None
        self.state_size = None
        self.action_size = None
        self.load_model()

    def update_state(self, guess, feedback):
        # Convert the guess to a number
        num_guess = [self.alphabet.index(letter) for letter in guess]
        print(num_guess)
        # Update the current state
        for i in range(len(num_guess)):
            if feedback[i] == 'green':
                self.current_state[i][num_guess[i]] = 2
            elif feedback[i] == 'yellow':
                self.current_state[i][num_guess[i]] = 1
            elif feedback[i] == 'gray' and np.array_equiv(self.current_state[:, num_guess[i]], np.zeros(5)):
                for j in range(len(guess)):
                    self.current_state[j][num_guess[i]] = 3
        # DEBUG
        print(self.current_state)
        self.current_guess_index += 1

    def update_word_list(self, guess, feedback):
        for word in self.filtered_word_list:
            valid = True

            green_indices = [i for i, f in enumerate(feedback) if f == 'green']
            green_letters = [guess[i] for i in green_indices]
            yellow_indices = [i for i, f in enumerate(feedback) if f == 'yellow']
            yellow_letters = [guess[i] for i in yellow_indices]
            gray_indices = [i for i, f in enumerate(feedback) if f == 'gray']

            # Check for green feedback
            for i in green_indices:
                if word[i] != guess[i]:
                    valid = False
                    break
            if not valid:
                continue
            # Check for gray feedback
            for i in gray_indices:
                if guess[i] in yellow_letters + green_letters:
                    continue
                if guess[i] in word:
                    valid = False
                    break
            if not valid:
                continue
            # Check for yellow feedback
            for i in yellow_indices:
                if (guess[i] not in word) or (guess[i] == word[i]):
                    valid = False
                    break

            if not valid:
                self.filtered_word_list.remove(word)

    def load_model(self):
        # Load the model if it exists
        if os.path.exists(self.file_name):
            # Load the model
            saved_model = torch.load(self.file_name)
            # Get the state and action sizes
            self.state_size = saved_model['input_size']
            self.action_size = saved_model['output_size']

            # Set up the model
            self.agent = DQAgent(self.state_size, self.action_size, training=False, file_name=self.file_name)
            print("Model loaded.")
        else:
            raise Exception("No model found.")

    def get_guess(self):
        # Find the index of the filtered words in the word list
        index_list = [self.word_list.index(value) for value in self.filtered_word_list if value in self.word_list]

        # Get the current state
        state = np.concatenate((self.current_state.flatten().copy() / 3, [self.current_guess_index / 6]))
        # Get the action values for the state
        action = self.agent.action_values(state)[0]

        print("best word:", self.word_list[np.argmax(action)])

        # Filter the action values to only include the filtered words
        filtered_action = np.zeros(len(action))
        for index in index_list:
            filtered_action[index] = action[index]

        # Get the index of the highest value
        action = np.argmax(filtered_action)
        return self.word_list[action]