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

    # load saved model
    saved_model = torch.load('wordle_dqn_model.pth')
    state_size = saved_model['input_size']

    action_size = saved_model['output_size']

    # Set up the model
    model = DQN(state_size, action_size)

    # Load the model if it exists
    if os.path.exists('wordle_dqn_model.pth'):
        # Ask the user if they want to load the model
        model.load_state_dict(saved_model["model_state"])
        print("Model loaded.")
    else:
        raise Exception("No model found.")
    model.eval()
    # Game loop
    try:
        while True:
            state = env.reset()
            no_word_vector = np.zeros(5)
            state = np.concatenate((no_word_vector, state, [1]))

            done = False
            # Print the target word
            print(env.one_hot_to_word(env.target_word))

            while not done:
                action = choose_action(state, action_size, 0, model)
                word_state = encode_guess(word_list[action])
                feedback_state, reward, done = env.step(action)  # Perform the action in the environment

                next_state = np.concatenate((word_state, feedback_state, [env.current_guess_index]))
                state = next_state

                # Print the guess and feedback
                print(f"Guess: {word_list[action]}")
                print(f"Feedback: {feedback_state}")

                # Sleep for 1 second
                time.sleep(1)


    except KeyboardInterrupt:
        print("Exiting...")
        exit()