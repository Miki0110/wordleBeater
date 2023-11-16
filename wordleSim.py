import random
import numpy as np


class WordleEnvironment:
    def __init__(self, word_list, alphabet='abcdefghijklmnopqrstuvwxyzæøåé'):
        assert all(len(word) == 5 for word in word_list), "All words must be 5 letters long"
        self.alphabet = alphabet
        self.words = word_list
        self.target_word = None
        self.current_state = None
        self.max_guesses = 6
        self.reset()

    def reset(self):
        self.target_word = random.choice(self.words)
        # Initialize a numeric state representation
        self.current_state = {
            'guesses': np.zeros((self.max_guesses, 5, len(self.alphabet))),
            'feedback': np.zeros(5)
        }
        self.current_guess_index = 0
        return self._get_current_state()

    def step(self, action):
        word = self.words[action]
        # action is a one-hot encoded 5-letter word guess
        feedback = self.get_feedback(word)
        self.current_state['guesses'][self.current_guess_index] = word
        self.current_state['feedback'] = feedback
        self.current_guess_index += 1
        reward = self.get_reward(feedback, word)
        done = self.is_done(feedback)
        return self._get_current_state(), reward, done

    def _get_current_state(self):
        # Flatten the guesses and feedback arrays and concatenate them
        #flat_guesses = self.current_state['guesses'].reshape(-1)
        #flat_feedback = self.current_state['feedback'].reshape(-1)
        return self.current_state['feedback']

    def get_feedback(self, guess):
        """
        Generates feedback for a guess.
        :param guess: The guessed word
        :param target_word: The target word
        :return: A list representing feedback (green=2, yellow=1, gray=0)
        """
        target_word = self.target_word
        feedback = np.zeros(5)  # Initialize feedback with all grays (0s)

        # Track letters in the target word that have been matched
        matched = [False] * len(target_word)

        # First pass for green feedback
        for i in range(len(guess)):
            if np.array_equiv(guess[i], target_word[i]):
                feedback[i] = 2  # Green
                matched[i] = True

        # Second pass for yellow feedback
        for i in range(len(guess)):
            if feedback[i] == 0:  # Only check letters not already marked green
                for j in range(len(target_word)):
                    if not matched[j] and np.array_equiv(guess[i], target_word[j]):
                        feedback[i] = 1  # Yellow
                        matched[j] = True
                        break

        return feedback

    def get_reward(self, feedback, action):
        # TODO: Improve the reward function
        # Initial reward based on correct positions and correct letters
        reward = sum(feedback)/100

        # Reward if the word is guessed correctly
        if np.array_equiv(action, self.target_word):
            reward += 1

        # Penalty for guessing the same word twice
        if self.current_guess_index > 1:
            if np.array_equiv(action, self.current_state['guesses'][self.current_guess_index-1]):
                reward -= 1
        return reward

    def is_done(self, feedback):
        # Check if the game is done (word is guessed correctly or max guesses reached)
        return feedback[1] == 5 or self.current_guess_index >= self.max_guesses

    # Conversion functions
    def _letter_to_one_hot(self, letter):
        one_hot = np.zeros(len(self.alphabet))
        one_hot[self.alphabet.index(letter)] = 1
        return one_hot

    def _one_hot_to_letter(self, one_hot):
        return self.alphabet[np.argmax(one_hot)]

    def word_to_one_hot(self, word):
        return np.array([self._letter_to_one_hot(letter) for letter in word])

    def one_hot_to_word(self, one_hot_word):
        return ''.join(self._one_hot_to_letter(one_hot_letter) for one_hot_letter in one_hot_word)

def convert_word_list(words, alphabet='abcdefghijklmnopqrstuvwxyzæøåé'):
    word_to_one_hot = lambda word: np.array([np.eye(len(alphabet))[alphabet.index(letter)] for letter in word])
    return [word_to_one_hot(word) for word in words]


if __name__ == '__main__':
    # Example usage
    with open('wordle_list.txt', 'r', encoding='UTF-8', newline='\r\n') as f:
        words = [line.strip().lower() for line in f]
    env = WordleEnvironment(convert_word_list(words))

    # Reset the environment to start
    initial_state = env.reset()
    target_word = env.one_hot_to_word(env.target_word)
    print(f"Target Word: {target_word}")  # For testing purposes only

    while True:
        # Start the environment
        guess_numerical = random.randint(0, len(words) - 1)
        guess_word = words[guess_numerical]
        next_state, reward, done = env.step(guess_numerical)

        # Extract the feedback for the latest guess from the next_state
        feedback = next_state[-2:]  # Assumes feedback is the last two elements of the state array
        print(f"Guess: {guess_word}, Feedback: {feedback}, Reward: {reward}, Done: {done}")

        if done:
            break