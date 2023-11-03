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
            'feedback': np.zeros((self.max_guesses, 2))
        }
        self.current_guess_index = 0
        return self._get_current_state()

    def step(self, action):
        word = self.words[action]
        # action is a one-hot encoded 5-letter word guess
        feedback = self.get_feedback(word)
        self.current_state['guesses'][self.current_guess_index] = word
        self.current_state['feedback'][self.current_guess_index] = feedback
        self.current_guess_index += 1
        reward = self.get_reward(feedback, word)
        done = self.is_done(feedback)
        return self._get_current_state(), reward, done

    def _get_current_state(self):
        # Flatten the guesses and feedback arrays and concatenate them
        flat_guesses = self.current_state['guesses'].reshape(-1)
        flat_feedback = self.current_state['feedback'].reshape(-1)
        return np.concatenate([flat_guesses, flat_feedback])

    def get_feedback(self, guess):
        # Compare the guess to the target word and return feedback
        feedback = {
            'correct_position': sum([1 for g, t in zip(guess, self.target_word) if np.array_equal(g, t)]),
            'correct_letter': sum([1 for g in guess if any(np.array_equal(g, t) for t in self.target_word)])
        }
        feedback['correct_letter'] -= feedback['correct_position']  # Subtract correct positions to correct the count
        feedback_array = np.array([feedback['correct_letter'], feedback['correct_position']])
        return feedback_array

    def get_reward(self, feedback, action):
        # Initial reward based on correct positions and correct letters
        reward = feedback[1] * 0.2 + feedback[0] * 0.1

        # Bonus for repositioning yellow letters correctly
        for prev_guess, prev_feedback in zip(self.current_state['guesses'], self.current_state['feedback']):
            for i, (prev_letter, prev_feedback) in enumerate(zip(prev_guess, prev_feedback)):
                if prev_feedback == 'yellow' and np.array_equal(prev_letter, self.target_word[i]):
                    reward += 0.1  # Bonus for correctly repositioning a yellow letter

        # Penalty for exceeding the guess limit or repeating the same incorrect guess
        if self.current_guess_index >= self.max_guesses:
            reward -= 10
        elif any(np.array_equal(action, prev_guess) for prev_guess in self.current_state['guesses'][:-1]):  # Updated line
            reward -= 0.1

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
    with open('../wordle_list.txt', 'r', encoding='UTF-8', newline='\r\n') as f:
        words = [line.strip().lower() for line in f]
    env = WordleEnvironment(convert_word_list(words))

    # Reset the environment to start
    initial_state = env.reset()
    target_word = env.one_hot_to_word(env.target_word)
    print(f"Target Word: {target_word}")  # For testing purposes only

    while True:
        # Start the environment
        guess_numerical = random.randint(0, len(words) - 1)
        guess_word = env.one_hot_to_word(words[guess_numerical])
        next_state, reward, done = env.step(guess_numerical)

        # Extract the feedback for the latest guess from the next_state
        feedback = next_state[-2:]  # Assumes feedback is the last two elements of the state array
        print(f"Guess: {guess_word}, Feedback: {feedback}, Reward: {reward}, Done: {done}")

        if done:
            break