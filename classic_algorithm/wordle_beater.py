from collections import Counter
import random


def get_startword():
    best_starting_words = ["bares", "bores", "boles", "sande", "salte", "serie", "sanke", "satse", "salme", "sadle"]
    return random.choice(best_starting_words)


# Check if a word is possible based on the feedback for a given guess
def filter_words(wordlist, guess, feedback):
    filtered_list = []

    for word in wordlist:
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

        if valid:
            filtered_list.append(word)

    return filtered_list


# Make a guess based on remaining words and past feedback
def make_guess(words, previous_feedback):
    if not words:
        return "....."

    all_letters = ''.join(words)
    letter_frequency = Counter(all_letters)

    word_scores = {}
    for word in words:
        word_scores[word] = 0

    past_guesses = [entry['guess'] for entry in previous_feedback]

    for feedback_entry in previous_feedback:
        word_scores = adjust_scores_based_on_feedback(word_scores, letter_frequency, feedback_entry)

    best_guess = max(word_scores, key=word_scores.get)
    return best_guess


def has_double_letters(word):
    """Check if the word has double letters."""
    for i in range(len(word) - 1):
        if word[i] == word[i+1]:
            return True
    return False

# Adjust scores based on each feedback
def adjust_scores_based_on_feedback(word_scores, letter_frequency, feedback_entry):
    guess = feedback_entry['guess']
    feedback = feedback_entry['feedback']

    for word, score in word_scores.items():
        # Penalize words with double letters
        if has_double_letters(word):
            word_scores[word] -= 1000
        for i, letter in enumerate(word):
            if feedback[i] == "green" and guess[i] == letter:
                word_scores[word] += 10 * letter_frequency[letter]
            elif feedback[i] == "gray" and guess[i] == letter:
                word_scores[word] -= 10 * letter_frequency[letter]
            elif feedback[i] == "yellow" and letter in guess and word[i] != guess[i]:
                word_scores[word] += 5 * letter_frequency[letter]
    return word_scores

if __name__ == '__main__':
    pass
