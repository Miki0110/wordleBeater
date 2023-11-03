
with open('../ordklasse.txt', 'r', encoding='UTF-8', newline='\r\n') as f:
    text = f.read()
# Split the words into a list
words = text.splitlines()

# Find the words with 5 letters and remove the semicolon
five_letter_words = [word.split(';')[0].split(" ")[-1] for word in words if len(word.split(';')[0].split(" ")[-1]) == 5]

# Remove words that contain special characters
five_letter_words = [word for word in five_letter_words if word.isalpha()]

# Save the words to a file
with open('../wordle_list.txt', 'w', encoding='UTF-8', newline='\r\n') as f:
    for word in five_letter_words:
        f.write(word + '\n')