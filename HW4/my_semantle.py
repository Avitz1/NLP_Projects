from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sys import argv
import random
import nltk
from nltk.corpus import brown
import json

# this function extract some info about the difficulty levels of the words in the brown corpus,
# by measuring how close the closest words are.
# this is used to suit the desired difficulty of the puzzle/word game
def preprocess_data(pre_trained_model):
    # import the words of the brown corpus
    words = brown.words()
    # make the corpus lowercase
    words = [word.lower() for word in words]
    words = set(words)
    # remove punctuation and numbers and http's from words
    words = [word for word in words if word.isalpha()]
    # remove words that are less than 3 characters long
    #words = [word for word in words if len(word) > 2 and len(word) < 16]
    # remove stopwords
    words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
    words = [word for word in words if word in pre_trained_model]
    words_level = {}
    for word in words:
        # get the most similar words
        most_similar = pre_trained_model.most_similar(word, topn=10)
        words_level[word] = (most_similar[0][1], most_similar[5][1], most_similar[9][1])
    with open("words_level.json", 'w') as f:
        json.dump(words_level, f)
    # divide the words_level to 2 levels
    level1 = {}
    level2 = {}
    for word in words_level:
        if words_level[word][0] > 0.75 or words_level[word][1] > 0.68 or words_level[word][2] > 0.60:
            level1[word] = words_level[word]
        else:
            level2[word] = words_level[word]

    # save the first dict as "easy.json" and the second as "hard.json"
    with open("easy.json", 'w') as f:
        json.dump(level1, f)
    with open("hard.json", 'w') as f:
        json.dump(level2, f)

# this function creats the key-value pairs for the word game
def create_kv(glove_file):
    # Save the GloVe text file as a word2vec file for your use:
    pre_trained_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    # Load the file as KeyVectors:
    pre_trained_model = KeyedVectors.load_word2vec_format("word2vec.txt", binary=False)
    pre_trained_model.save("key_vectors.kv")

# this function runs the word game
def run_game(pre_trained_model):
    # ------ hello message ------
    print("Welcome to the word game!")
    while True:
        # ------ choose difficulty level ------
        print("Choose the difficulty level:")
        print("1. Easy")
        print("2. Hard")
        level = input("Enter 1 or 2: ")
        # check if the input is valid
        while level != "1" and level != "2":
            level = input("Invalid level, enter 1 or 2: ")
        tries = input("Is the sky your limit? Enter the number of tries: ")
        # check if the input is valid
        while not tries.isdigit() or int(tries) < 1:
            tries = input("Invalid number, enter a number: ")
        guesses = 0
        hints = 0
        best_score = 0
        similiarity = 0
        # enumarate the levels
        levels = {1: "easy.json", 2: "hard.json"}
        # load the corresponding json file
        with open(levels[int(level)], 'r') as f:
            words = json.load(f)
        # choose a random word from the json file
        word = random.choice(list(words.keys()))
        print("We have chosen a word for you. Try to guess it!")
        while guesses < int(tries):
            # ------ hint ------
            hint = input("\nDo you want a hint? Enter y/n: ")
            # check if the input is valid
            while hint != "y" and hint != "n":
                hint = input("Invalid input, enter y/n: ")
            if hint == "y":
                # get the most similar words
                most_similar = pre_trained_model.most_similar(word, topn=100)
                if best_score != similiarity:
                    best_score = max(best_score * 1.02, similiarity, most_similar[99][1])
                # print the first word which is 10% more similiar to the word from the last guess
                for i in range(99, 0, -1):
                    if most_similar[i][1] > best_score * 1.02:
                        print("A hint: {}".format(most_similar[i][0]))
                        print("The similarity between the word and the hint is:   {:.5f}".format(most_similar[i][1]))
                        break
                hints += 1
            # ------ guess the word ------
            guess = input("\nEnter your guess: ")
            # check if the guess is valid, meaning it is a word which has a key vector
            while guess not in pre_trained_model:
                guess = input("Invalid guess, please enter a word: ")

            # ------ check if the guess is correct ------
            # if the guess is correct, print a message with the number of guesses and break the loop
            if guess == word:
                print("Congratulations! You guessed the word in {} tries!".format(guesses))
                break
            # if the guess is incorrect, print a message with the number of guesses and let the user try again
            else:
                # check how similar the guess is to the word
                similiarity = pre_trained_model.similarity(word, guess)
                # if the similarity is above 0.5, print a message with the similarity (only 5 digits after the dot)
                if similiarity > best_score:
                    print("The similarity between the word and your guess is:  {:.5f}".format(similiarity))
                    print("you are getting closer!\n")
                # if the similarity is below 0.5, print a message with the similarity
                else:
                    print("The similarity between the word and your guess is: {:.5f}.".format(similiarity))
                    print("you are still far!\n")
                guesses += 1
                best_score = max(best_score, similiarity)
            # ------ game over? ------
            # if the user has exceeded the number of tries, print a message with the correct word
            if guesses == int(tries):
                print("You have exceeded the number of tries! The word was:  --{}--".format(word))
                break
            # ask the user if they want to proceed with the guesses
            proceed = input("Do you want to proceed with the guesses? Enter y/n: ")
            # check if the input is valid
            while proceed != "y" and proceed != "n":
                proceed = input("Invalid input, enter y/n: ")
            if proceed == "y":
                continue
            else:
                break
        # ask the user if they want to play again
        play_again = input("\nDo you want to play again with a new puzzle? Enter y/n: ")
        if play_again == "y":
            continue
        else:
            break
    # ------ goodbye message ------
    print("\nThank you for playing!")

if __name__ == "__main__":
    if len(argv) != 2:
        kv_file = "key_vectors.kv"
    else:
        kv_file = argv[1]
    pre_trained_model = KeyedVectors.load(kv_file, mmap='r')
    # to process the GloVe file and create the json files (takes a while) uncomment the following lines:
    # preprocess_data(pre_trained_model)
    # create_kv("glove.6B.300d.txt")
    # pre_trained_model = KeyedVectors.load("key_vectors.kv", mmap='r')
    run_game(pre_trained_model)








