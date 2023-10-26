import random
import sys
import os
sys.path.append('../../../..')

from Wrong_word_generator import wrong_word_generator


def sentence_generation(sentence: str) -> None:
    """
    Input: single sentence
           num_of_wrong_sentence: how many wrong sentences are generated
    
    Output: pair of correct & incorrect sentence

    Example:
        inp: correct sentence
        out: correct sentence|incorrect sentence_1
             correct sentence|incorrect sentence_2
             correct sentence|incorrect sentence_3
             correct sentence|incorrect sentence_4
             correct sentence|incorrect sentence_5
    
    Constraints:
        len(sen) > 20: sai 5 words
        len(sen) > 10: sai 2 words

    Naming convention:
        sentence_0.txt
        sentence_1.txt
        sentence_2.txt
    """
    ['']
    if len(sentence) <= 10:
        max_wrong_words = 3
    elif 10 < len(sentence) <= 20:
        max_wrong_words = 5
    elif len(sentence) > 20:
        max_wrong_words = 7
        
    for i in range(num_of_wrong_sentence):
        correct_sentence = sentence
        incorrect_sentence = create_incorrect_sentence(correct_sentence, max_wrong_words)
        print(f"{correct_sentence}|{incorrect_sentence}")


    
    return None

def main() -> None:
    sentence_generation()
    return None


if __name__ == '__main__':
    main()






import random
import sys
import os
sys.path.append('../../../..')

from Wrong_word_generator import wrong_word_generator



def sentence_generation(sentence: str, num_of_wrong_sentence: int) -> None:
    """
    Input: single sentence
           num_of_wrong_sentence: how many wrong sentences are generated
    
    Output: pair of correct & incorrect sentence

    Example:

    
    Constraints:
        len(sen) > 20: sai 5 words
        len(sen) > 10: sai 2 words

    Naming convention:        inp: correct sentence
        out: correct sentence|incorrect sentence_1
             correct sentence|incorrect sentence_2
             correct sentence|incorrect sentence_3
             correct sentence|incorrect sentence_4
             correct sentence|incorrect sentence_5
        sentence_0.txt
        sentence_1.txt
        sentence_2.txt
    """
    if len(sentence) <= 10: #>7
        max_wrong_words = 3
    elif 10 < len(sentence) <= 20:
        max_wrong_words = 5
    else:
        max_wrong_words = 3  # No errors if sentence length >20

    for i in range(num_of_wrong_sentence):
        correct_sentence = sentence
        incorrect_sentence = create_incorrect_sentence(correct_sentence, max_wrong_words)
        print(f"{correct_sentence}|{incorrect_sentence}")

def create_incorrect_sentence(sentence, max_wrong_words):
    generator = wrong_word_generator.wrong_word_generator()
    words = sentence.split()
    incorrect_sentence = sentence
    num_wrong_words = random.randint(1, max_wrong_words)  # Randomly choose how many words to make wrong
    incorrect_words = []
    for _ in range(num_wrong_words):
        index = random.randint(0, len(words) - 1)
        word = words[index]
        generator.set_word(word)
        incorrect_word_dict = generator.insert_char_english()
        incorrect_word_list = list(incorrect_word_dict.values())[0].split(', ')  # Tạo danh sách các từ sai từ từ điển
        incorrect_word = random.choice(incorrect_word_list)
        incorrect_sentence = incorrect_sentence.replace(word, incorrect_word, 1)
        #words[index] = incorrect_word
    return incorrect_sentence
            # insert(words[index])  # Create errors for the selected word


def main() -> None:
    sentence = "Three boys are playing football in the playground"
    max_wrong_words = 5
    sentence_generation(sentence, max_wrong_words)

    # corpus_file = 'corpus.txt'
    # output_folder = 'output_sentences'
    # os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    #
    # with open(corpus_file, 'r') as file:
    #     sentences = file.read().splitlines()
    #
    # for i, sentence in enumerate(sentences):
    #     output_file = os.path.join(output_folder, f'sentence_{i}.txt')
    #     with open(output_file, 'w') as file:
    #         file.write(f'{sentence}|{"|".join(sentence_generation(sentence, 5))}')
    #
    return None

if __name__ == '__main__':
    main()



