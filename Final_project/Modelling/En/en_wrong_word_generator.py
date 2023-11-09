from math import ceil
import string
import random
import numpy as np


class WrongWordGenerator:
    def __init__(self):
        self.__word = ""


    def get_word(self) -> str:
        return self.__word


    def set_word(self, word: str):
        self.__word = word


    def remove_char(self) -> str:
        word = self.__word
        if len(word) == 1:
            return ""
        elif len(word) > 1:
            pos = random.randrange(0, len(word))
            # remove char
            word = word.replace(word[pos], "")
            return word


    def insert_char(self, alphabet=string.ascii_lowercase) -> str:
        word = self.__word

        pos_1 = random.randrange(0, len(word))
        pos_2 = random.randrange(0, len(alphabet))  # select random char

        # insert random char into word
        word = word[:pos_1] + alphabet[pos_2] + word[pos_1:]
        return word


    def swap_char(self) -> str:
        word = self.__word

        pos_1 = random.randrange(0, len(word))
        pos_2 = random.randrange(0, len(word))

        if pos_1 == pos_2:
            return word
        else:
            # swap char
            tmp = word[pos_1]
            word = word.replace(word[pos_1], word[pos_2], 1)
            word = word.replace(word[pos_2], tmp, 1)
        return word


###############################################################################
def add_noise(sequence: str, language: str, noise_rate=0.2, num_of_methods=3) -> str:
    sequence = sequence.strip().split(" ")
    sentence_len = len(sequence)
    max_wrong_words = int(np.ceil(sentence_len * noise_rate))

    wrong_word_generator = WrongWordGenerator()
    for i in range(max_wrong_words):
        # random word
        word_index = random.randrange(0, sentence_len)
        randomized_word = sequence[word_index]
        wrong_word_generator.set_word(randomized_word)

        # randomize method that's gonna be applied
        method_prob = np.random.uniform(low=0.0, high=1.0)
        if method_prob < 1/num_of_methods:
            # no noise
            generated_result = randomized_word
        elif 1/num_of_methods <= method_prob < 2/num_of_methods:
            # remove
            generated_result = wrong_word_generator.remove_char()
        elif 2/num_of_methods <= method_prob < 3/num_of_methods:
            # insert
            generated_result = wrong_word_generator.insert_char()
        elif 3/num_of_methods <= method_prob <= 1:
            # swap
            generated_result = wrong_word_generator.swap_char()
        sequence[i] = generated_result
    return " ".join(sequence)