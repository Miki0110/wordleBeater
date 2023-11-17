import numpy as np
import torch
import os
from wordleSim import WordleEnvironment
from model import DQN, choose_action, encode_guess, read_word_list
import time


if __name__ == "__main__":
    # Set up the sim environment
    word_list = read_word_list()[50:150]
    env = WordleEnvironment(word_list)

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
            state = env.reset()[0].copy() / 3  # Reset the Wordle environment
            remaining_guesses_state = 1 / 6
            state = np.concatenate((state.flatten(), [remaining_guesses_state]))

            done = False
            # Print the target word
            print(env.numbers_to_word(env.target_word))

            while not done:
                action = choose_action(state, action_size, 0, model)
                feedback, reward, done = env.step(action)  # Perform the action in the environment
                feedback_state = feedback[0].copy() / 3
                remaining_guesses_state = env.current_guess_index / 6

                next_state = np.concatenate((feedback_state.flatten(), [remaining_guesses_state]))

                state = next_state

                # Print the guess and feedback
                print(f"Guess: {word_list[action]}")
                print(f"Feedback: {feedback[1][-1]}")

                # Sleep for 1 second
                time.sleep(1)


    except KeyboardInterrupt:
        print("Exiting...")
        exit()