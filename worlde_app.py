import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from wordle_beater import filter_words, make_guess, get_startword


class FeedbackLabel(QLabel):
    """ Label used to describe the feedback for a letter in the word """
    def __init__(self, *args, **kwargs):
        super(FeedbackLabel, self).__init__(*args, **kwargs)
        self.feedback_states = ["gray", "yellow", "green"]
        self.current_feedback = 0

    def mousePressEvent(self, event):
        # Cycle through the feedback states, when clicked
        self.current_feedback = (self.current_feedback + 1) % 3
        self.update_feedback_style()

    def update_feedback_style(self):
        # Update the style of the label, based on the current feedback state
        color = self.feedback_states[self.current_feedback]
        if color == "green":
            self.setStyleSheet("background-color: green; border: 1px solid black; width: 40px; height: 40px; font-size: 20px;")
        elif color == "yellow":
            self.setStyleSheet("background-color: yellow; border: 1px solid black; width: 40px; height: 40px; font-size: 20px;")
        else:
            self.setStyleSheet("background-color: gray; border: 1px solid black; width: 40px; height: 40px; font-size: 20px;")


class WordleGame(QWidget):
    """ Main window for the Wordle game """
    def __init__(self):
        super().__init__()
        # Read the word list
        with open('wordle_list.txt', 'r', encoding='UTF-8', newline='\r\n') as f:
            words = [line.strip() for line in f]
        self.words = words
        # Initialize the game
        self.current_attempt = 0
        # Feedback is saved as a list of strings, where each string is either "gray", "yellow" or "green"
        # EG: ["gray", "yellow", "green", "gray", "gray"]
        self.previous_feedback = ['gray'] * 5

        # Initialise the first guess, based on the best starting words
        self.current_guess = make_guess([get_startword()])

        self.init_ui()

    def init_ui(self):
        # Create the layout
        self.layout = QVBoxLayout()

        self.letter_layout = QHBoxLayout()
        # Create the labels for the letters
        self.letters = [FeedbackLabel(self.current_guess[i].upper(), self) for i in range(5)]
        # Add the labels to the layout
        for letter in self.letters:
            letter.setAlignment(Qt.AlignCenter)
            letter.setStyleSheet("background-color: gray; border: 1px solid black; width: 40px; height: 40px; font-size: 20px;")
            self.letter_layout.addWidget(letter)
        self.layout.addLayout(self.letter_layout)

        # Create the submit button
        self.submit_btn = QPushButton("Submit Feedback", self)
        self.submit_btn.clicked.connect(self.submit_feedback)
        self.layout.addWidget(self.submit_btn)

        self.setLayout(self.layout)
        self.setWindowTitle("Wordle Game")
        self.show()

    def make_guess(self):
        # Use the make_guess function from wordle_beater.py
        if self.words:
            return make_guess(self.words)  # Use make_guess from wordle_beater
        else:
            return "....."  # Return a placeholder if no words left

    def submit_feedback(self):
        # Get the feedback from the labels
        feedback = [self.determine_feedback(label) for label in self.letters]
        self.previous_feedback = feedback  # Store the feedback
        # Check if the word was found
        if feedback == ['green'] * 5:
            self.end_game("Word found!")
            return
        # Filter the words based on the feedback
        self.words = filter_words(self.current_guess, feedback, self.words)
        self.current_attempt += 1
        # Check if the game is over, because no words are left
        if not self.words:
            self.end_game("Failed to guess the word!")
            return
        # Make a new guess
        self.current_guess = self.make_guess()
        self.set_next_guess_and_feedback()

    def set_next_guess_and_feedback(self):
        # Update the labels with the new guess and feedback
        for i, letter in enumerate(self.letters):
            letter.setText(self.current_guess[i].upper())
            # If it was green before, it should stay green
            if self.previous_feedback[i] == "green":
                letter.setStyleSheet("background-color: green; border: 1px solid black; width: 40px; height: 40px; font-size: 20px;")
            else:
                letter.setStyleSheet("background-color: gray; border: 1px solid black; width: 40px; height: 40px; font-size: 20px;")
                # Reset the feedback back to gray
                letter.current_feedback = 0

    def determine_feedback(self, label):
        return label.feedback_states[label.current_feedback]

    def end_game(self, message):
        self.submit_btn.setDisabled(True)
        for label in self.letters:
            label.setContextMenuPolicy(Qt.NoContextMenu)
        print(message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = WordleGame()
    sys.exit(app.exec_())
