import numpy as np
import torch
import os
from wordleSim import WordleEnvironment, convert_word_list
from model import DQN, choose_action, encode_guess, read_word_list, train_model
import time


if __name__ == "__main__":
    # Set up the sim environment
    word_list = read_word_list()
    env = WordleEnvironment(convert_word_list(word_list))

    # Calculate the state vector size
    alphabet = 'abcdefghijklmnopqrstuvwxyzæøåé'
    one_hot_length = len(alphabet)
    word_state_length = one_hot_length * 5
    feedback_state = 5
    state_size = word_state_length + feedback_state

    action_size = len(word_list)

    # Set up the model
    model = DQN(state_size, action_size)

    # Load the model if it exists
    if os.path.exists('wordle_dqn_model.pth'):
        # Ask the user if they want to load the model
        model.load_state_dict(torch.load('wordle_dqn_model.pth'))
        print("Model loaded.")
    else:
        raise Exception("No model found.")

    # Game loop
    try:
        while True:
            state = env.reset()
            no_word_vector = np.zeros(word_state_length)
            state = np.concatenate((no_word_vector, state))

            done = False
            # Print the target word
            print(env.one_hot_to_word(env.target_word))

            while not done:
                action = choose_action(state, action_size, 0, model)
                word_state = encode_guess(word_list[action])
                feedback_state, reward, done = env.step(action)  # Perform the action in the environment

                next_state = np.concatenate((word_state, feedback_state))
                state = next_state

                # Print the guess and feedback
                print(f"Guess: {word_list[action]}")
                print(f"Feedback: {feedback_state}")

                # Sleep for 1 second
                time.sleep(1)


    except KeyboardInterrupt:
        print("Exiting...")
        exit()