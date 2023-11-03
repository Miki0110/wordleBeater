import random
import numpy as np


class WordleEnvironment:
    def __init__(self, word_list):
        self.word_list = word_list
        self.target_word = None
        self.current_guess = None
        self.attempts = 0
        self.max_attempts = 6
        self.done = False
        self.feedback_history = []

    def reset(self):
        self.target_word = random.choice(self.word_list)
        self.attempts = 0
        self.done = False
        self.current_guess = None
        self.feedback_history = []
        # Reset state and return initial state
        return self.get_state()

    def step(self, guess):
        self.current_guess = guess
        self.attempts += 1

        feedback = self.get_feedback()
        self.feedback_history.append(self.encode_feedback(feedback))

        # Check if the guess is correct
        if guess == self.target_word:
            self.done = True
            reward = 100  # Reward for correct guess
        elif self.attempts >= self.max_attempts:
            self.done = True
            reward = -10  # Penalty for not guessing in time
        else:
            reward = self.calculate_reward(guess)  # Calculate intermediate reward

        return self.get_state(), reward, self.done

    def calculate_reward(self, guess):
        # A simple reward function that gives 1 point for each correct letter
        return sum(1 for g, t in zip(guess, self.target_word) if g == t)

    def get_feedback(self):
        # This function returns the feedback similar to Wordle's green, yellow, and gray
        feedback = []
        for i, letter in enumerate(self.current_guess):
            if letter == self.target_word[i]:
                feedback.append('green')
            elif letter in self.target_word:
                feedback.append('yellow')
            else:
                feedback.append('gray')
        return feedback

    def get_state(self):
        if self.current_guess:
            # Convert letters to indices
            letter_indices = [self.letter_to_index(letter) for letter in self.current_guess]
            feedback = self.get_feedback()
            feedback_data = self.encode_feedback(feedback)
        else:
            # If there's no current guess, use -1 as a placeholder for the letter index
            letter_indices = np.zeros(5)
            feedback_data = np.zeros((3 * 5))  # Assuming 3 possible feedback states for each character

        state = {
            'letter_indices': np.array(letter_indices),
            'feedback_data': feedback_data
        }
        return state

    def letter_to_index(self, letter):
        # Mapping from letter to index, accounting for the additional Danish characters
        letter_mapping = {
            'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4,
            'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
            'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14,
            'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19,
            'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24,
            'z': 25, 'æ': 26, 'ø': 27, 'å': 28
        }
        return letter_mapping[letter]

    def encode_feedback(self, feedback):
        # Simplified encoding with 3 slots per character
        encoding = {'green': [1, 0, 0], 'yellow': [0, 1, 0], 'gray': [0, 0, 1]}
        encoded_feedback = [encoding[color] for color in feedback]
        return np.array(encoded_feedback).flatten()

    def is_done(self):
        return self.done


if __name__ == '__main__':
    # Example usage
    with open('../wordle_list.txt', 'r', encoding='UTF-8', newline='\r\n') as f:
        words = [line.strip() for line in f]
    env = WordleEnvironment(words)

    # Reset the environment to start
    initial_state = env.reset()
    print(f"Target Word: {env.target_word}")  # For testing purposes only

    while True:
        # Pick a random word
        word = random.choice(words)
        words.pop(words.index(word))
        print(f"Guess: {word}")
        # Example step (assuming the agent guessed the word "arise")
        next_state, reward, done = env.step(word)
        print(f"Feedback: {env.get_feedback()}")
        print(f"Reward: {reward}")
        print(f"Game Over: {done}")
        state = env.get_state()
        print(f"State: {state}")
        if done:
            break