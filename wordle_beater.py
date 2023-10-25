import random


def get_startword():
    best_starting_words = ["bares", "bores", "boles", "sande", "salte", "serie", "sanke", "satse", "salme", "sadle"]
    return random.choice(best_starting_words)

def filter_words(guess, feedback, words):
    new_words = []

    for word in words:
        match = True
        for i, (g, f) in enumerate(zip(guess, feedback)):
            if f == "green" and word[i] != g:
                match = False
                break
            if f == "gray" and word[i] == g:
                match = False
                break
            if f == "yellow" and word[i] == g:
                match = False
                break
            if f == "yellow" and g not in word:
                match = False
                break
        if match:
            new_words.append(word)
    return new_words


def make_guess(words):
    # TODO: Implement a smarter algorithm
    return words[0] if words else None


if __name__ == '__main__':
    pass
