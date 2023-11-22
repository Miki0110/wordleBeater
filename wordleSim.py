import random
import numpy as np


class WordleEnvironment:
    def __init__(self, word_list, alphabet='abcdefghijklmnopqrstuvwxyzæøåé'):
        assert all(len(word) == 5 for word in word_list), "All words must be 5 letters long"
        self.alphabet = alphabet
        self.words = self.convert_list_to_numbers(word_list)
        self.target_word = None
        self.current_state = np.zeros((5, len(self.alphabet)))
        self.guesses = []
        self.feedbacks = []
        self.max_guesses = 6
        self.current_guess_index = 0
        self.reset()

    def reset(self):
        self.target_word = random.choice(self.words)
        # Reset the current states
        self.current_state = np.zeros((5, len(self.alphabet)))
        self.guesses = []
        self.feedbacks = []
        self.current_guess_index = 0
        return self._get_current_state()

    def step(self, action):
        word = self.words[action]

        # Update the current state
        self.guesses.append(word)
        feedback = self.get_feedback(word)

        self.current_guess_index += 1
        reward = self.get_reward(feedback, word)
        done = self.is_done(feedback)
        return self._get_current_state(), reward, done

    def _get_current_state(self):
        return self.current_state, self.feedbacks

    def get_feedback(self, guess):
        target = self.target_word
        feedback = np.zeros(5)  # Initialize feedback with all grays (0s)
        used_in_target = [False] * len(target)

        # Check for green matches
        for i in range(len(guess)):
            if guess[i] == target[i]:
                feedback[i] = 2
                used_in_target[i] = True
                # Update the state
                self.current_state[i][guess[i]] = 2

        # Check for yellow matches
        for i in range(len(guess)):
            if feedback[i] == 0:
                for j in range(len(target)):
                    if not used_in_target[j] and guess[i] == target[j]:
                        feedback[i] = 1
                        used_in_target[j] = True
                        # Update the state
                        self.current_state[i][guess[i]] = 1
                        break

        # Mark remaining as no match
        for i in range(len(feedback)):
            if (feedback[i] == 0) and (guess[i] not in target):
                for j in range(5):
                    self.current_state[j][guess[i]] = 3
        self.feedbacks.append(feedback)
        return feedback

    def get_reward(self, feedback, action):
        # Initial reward based on correct positions and correct letters
        reward = sum(feedback)/10
        if not np.array_equiv(feedback, np.array([2, 2, 2, 2, 2])):
            # Penalty for incorrect letters
            reward -= 0.1 * self.current_guess_index

        # Reward if the word is guessed correctly
        if np.array_equiv(action, self.target_word):
            reward += 1*(self.max_guesses - self.current_guess_index)

        # Penalty for guessing the same word twice
        if self.current_guess_index > 1:
            if np.array_equiv(action, self.guesses[-2]):
                reward -= 1
        return reward

    def is_done(self, feedback):
        # Check if the game is done (word is guessed correctly or max guesses reached)
        return np.array_equiv(feedback, np.array([2, 2, 2, 2, 2])) or self.current_guess_index >= self.max_guesses

    # Conversion functions
    def _letter_to_number(self, letter):
        number = self.alphabet.index(letter)
        return number

    def _number_to_letter(self, number):
        letter = self.alphabet[number]
        return letter

    def word_to_numbers(self, word):
        return [self._letter_to_number(letter) for letter in word]

    def numbers_to_word(self, numbers):
        return ''.join([self._number_to_letter(number) for number in numbers])

    def convert_list_to_numbers(self, word_list):
        return [self.word_to_numbers(word) for word in word_list]


if __name__ == '__main__':
    # Example usage
    with open('wordle_list.txt', 'r', encoding='UTF-8', newline='\r\n') as f:
        words = [line.strip().lower() for line in f]
    env = WordleEnvironment(words)

    # Reset the environment to start
    initial_state = env.reset()
    target_word = env.numbers_to_word(env.target_word)
    print(f"Target Word: {target_word}")  # For testing purposes only

    total_reward = 0
    runs = 0
    while True:
        runs += 1
        # Start the environment
        guess_numerical = random.randint(0, len(words) - 1)
        if runs == 3:
            guess_numerical = words.index(target_word)
        guess_word = words[guess_numerical]
        next_state, reward, done = env.step(guess_numerical)

        total_reward += reward

        # Extract the feedback for the latest guess from the next_state
        feedback = next_state[1][-1]
        state = next_state[0]
        print(f"Guess: {guess_word}, Feedback: {feedback}, Reward: {reward}, Done: {done}")

        if done:
            print(f"Average Reward: {total_reward/runs}")
            break