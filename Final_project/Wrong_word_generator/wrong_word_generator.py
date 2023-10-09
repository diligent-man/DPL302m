import random
import string

class wrong_word_generator:
    def __init__(self):
        self.__word = None


    def get_word(self):
        return self.__word


    def set_word(self, word: str):
        self.__word = word


    def remove_char_english(self) -> dict:
        '''
        Input
            word: Input word to process

        Output
            result: dict that key is the original word,
                    value is the string of wrong words separated by delimiter ', '
        Example:
            inp: "hello"
            out: {"hello": "ello, hllo, helo, hell, hell"}
        '''
        word = self.get_word()
        result = {word: ''}

        # remove char in each pos
        for pos in range(len(word)):
            removed_word = word[: pos] + word[pos + 1:]
            result[word] += removed_word + ', '

        # Remove the trailing comma and space
        result[word] = result[word][: -2]
        return result


    def insert_char_english(self) -> dict:
        '''
        Input
            word: Input word to process

        Output
            result: dict that key is the original word,
                    value is the string of wrong words separated by delimiter ', '

        Example:
        words_each_pos = 2
            inp: "hi"
            out: {"hi": "ahi, bhi, chi, dhi, ......"}
        '''
        word = self.get_word()
        result = {word: ''}
        alphabet = string.ascii_lowercase  # Get all lowercase letters
        
        # Loop through each char pos
        for pos in range(len(word)+1):
            inserted_word = []
            # Insert at first pos
            if pos == 0:
                inserted_word = [char + word for char in alphabet]
            # insert at last pos
            elif pos == len(word):
                inserted_word = [word + char for char in alphabet]
            # insert other pos
            else:
                inserted_word = [word[:pos] + char + word[pos:] for char in alphabet]
            
            # merge words into a string
            inserted_word = (', ').join(inserted_word)

            # merge into result
            if pos == len(word):
                result[word] += inserted_word
            else:
                result[word] += inserted_word + ', '
        return result


    def swap_char_english(self) -> dict:
        """
        Input: word

        Output: dict that key is the original word,
                value is the string of wrong words separated by delimiter ', '
        
        Example
            inp: hello
            out: {"hello": }
        
        hello
        
        ohell, lohel, llohe, elloh, hello
        
        """


def test_remove_char_english() -> None:
    word = 'hello'
    generator = wrong_word_generator()
    generator.set_word(word)
    wrong_word = generator.remove_char_english()
    print(wrong_word)
    return None


def test_insert_char_english() -> None:
    word = 'hello'
    generator = wrong_word_generator()
    generator.set_word(word)
    wrong_word = generator.insert_char_english()
    print(wrong_word)
    

def test_swap_char_english() -> None:
    word = 'hello'
    generator = wrong_word_generator()
    generator.set_word(word)
    wrong_word = generator.swap_char_english()
    print(wrong_word)

def main() -> None:
    # test_remove_char_english()
    # test_insert_char_english()
    return None


if __name__ == '__main__':
    main()
