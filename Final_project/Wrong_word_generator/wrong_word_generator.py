"""
Structure:
wrong_word_generator
    # For both
    remove_char()

    # For En
    insert_char_english()
    swap_char_english()

    # For Vn
     wrong_tone_pos_vn()


Telex Reference:
    1/ https://vi.wikipedia.org/wiki/Quy_t%E1%BA%AFc_%C4%91%E1%BA%B7t_d%E1%BA%A5u_thanh_c%E1%BB%A7a_ch%E1%BB%AF_Qu%E1%BB%91c_ng%E1%BB%AF
    
    2/ https://gist.github.com/nguyenvanhieuvn/72ccf3ddf7d179b281fdae6c0b84942b?fbclid=IwAR2jH7XiK_nUeOO0muigSfHwIqkHq_pt7rE2UfO_vaP3AGhUGy03-a3sxq0

Telex rule: Telex requires the user to type in a base letter, followed by one or two characters that represent the diacritical marks:
"""
# sys.path.append('../../../..')
import os
import re
import string
import random
import zipfile
from itertools import count, permutations

from numpy import corrcoef


tonal_mapping_1 = {
    'ảo': ['aỏ'], 'áo': ['aó'], 'ào': ['aò'], 'ạo': ['aọ'], 'ão': ['aõ'],
    'ải': ['aỉ'], 'ái': ['aí'], 'ài': ['aì'], 'ại': ['aị'], 'ãi': ['aĩ'],
    'ảu': ['aủ'], 'áu': ['aú'], 'àu': ['aù'], 'ạu': ['aụ'], 'ãu': ['aũ'],
    'ảy': ['aỷ'], 'áy': ['aý'], 'ày': ['aỳ'], 'ạy': ['aỵ'], 'ãy': ['aỹ'],
    'òa': ['oà'], 'óa': ['oá'], 'ỏa': ['oả'], 'õa': ['oã'], 'ọa': ['oạ'],
    'oái': ['óai', 'oaí'], 'oài': ['òai', 'oaì'], 'oải': ['ỏai', 'oaỉ'], 'oại': ['ọai', 'oaị'],'oãi': ['õai', 'oaĩ'],
    'oáy': ['óay', 'oaý'], 'oạy': ['ọay', 'oaỵ'],
    'oạc': ['ọac'], 'oác': ['óac'],
    'oắc': ['óăc'], 'oặc': ['ọăc'],
    'oằm': ['òăm'],
    'oắn': ['óăn'], 'oằn': ['òăn'], 'oắn': ['óăn'], 'oẳn': ['ỏăn'],
    'oằng': ['òăng'], 'oẵng': ['õăng'], 'oắng': ['óăng'],
    'oắt': ['óăt'], 'oặt': ['ọăt'],
    'uần': ['ùân'], 'uẩn': ['ủân'], 'uẫn': ['ũân'], 'uận': ['ụân'],
    'uặng': ['ụặng'], 'uầng': ['ùâng'], 'uẫng': ['ũâng'], 'uẩng': ['ủâng'],
    'uật': ['ụât'], 'uất': ['úât'],
    'ẩu': ['âủ'], 'ấu': ['âú'], 'ầu': ['âù'], 'ậu': ['âụ'], 'ẫu': ['âũ'],
    'ẩy': ['âỷ'], 'ấy': ['âý'], 'ầy': ['âỳ'], 'ậy': ['âỵ'], 'ẫy': ['âỹ'],
    'uẩy': ['uâỷ', 'ủây'], 'uấy': ['uâý', 'úây'],   'uẫy': ['uâỹ', 'ũây'], 'uậy':['ụây','uâỵ'],
    'òe': ['oè'], 'óe': ['oé'], 'ỏe': ['oẻ'], 'õe': ['oẽ'], 'ọe': ['oẹ'],
    'oèn': ['òen'], 'oẻn': ['ỏen'],
    'oèo': ['òeo', 'oeò'], 'oẹo': ['ọeo', 'oẹo'], 'oẻo': ['ỏeo', 'oeỏ'],
    'éo': ['eó'], 'èo': ['eò'], 'ẹo': ['eọ'], 'ẻo': ['eỏ'], 'ẽo': ['eõ'],
    'oét': ['óet'], 'oẹt': ['ọet'],
    'uệ': ['ụê'], 'uế': ['úê'], 'uễ': ['ũê'], 'uề': ['ùê'], 'uể': ['ủê'],
    'uệch': ['ụêch'],
    'uềnh': ['ùênh'], 'uếnh': ['úênh'], 'uệnh': ['ụênh'],
    'ều': ['êù'], 'ếu': ['êú'], 'ệu': ['êụ'], 'ễu': ["êũ"], 'ểu': ['êủ'],
    'ùy': ['uỳ'], 'úy': ['uý'], 'ủy': ['uỷ'], 'uỵ': ['ụy'],
    'ìa': ['ià'], 'ĩa': ['iã'], 'ỉa': ['iả'], 'ía': ['iá'], 'ịa': ['iạ'],
    'uỵch': ['ụych'], 'uých': ['úych'],
    'iếc': ['íêc'], 'iệc': ['ịêc'],
    'iềm': ['ìêm'], 'iểm': ['ỉêm'], 'iếm': ['íêm'], 'iệm': ['ịêm'],
    'yểm': ['ỷêm'], 'yềm': ['ỳêm'], 'yếm': ['ýêm'], 'yệm': ['ỵêm'],
    'yến': ['ýên'],
    'iền': ['ìên'], 'iến': ['íên'], 'iễn': ['ĩên'], 'iển': ['ỉên'], 'iện': ['ịên'],
    'uyến': ['úyên', 'uýên'], 'uyền': ['ùyê', 'uỳên'], 'uyển': ['ủyên', 'uỷên'], 'uyện': ['ụyên', 'uỵên'], 'uyễn':['ũyên', 'uỹên'],
    'iếu': ['íêu', 'iêú'], 'iều': ['ìêu', 'iêù'], 'iệu': ['ịêu', 'iêụ'], 'iểu': ['ỉêu', ' iêủ'],
    'iết': ['íêt'], 'iềt': ['ìêt'], 'iệt': ['ịêt'],
    'iếng': ['íêng'], 'iềng': ['ìêng'], 'iệng': ['ịêng'], 'iểng': ['ỉêng'],
    'iếp': ['íêp'], 'iềp': ['ìêp'], 'iệp': ['ịêp'],
    'iết': ['íêt'], 'iềt': ['ìêt'], 'iệt': ['ịêt'],
    'uyệt': ['ụyêt', 'uỵêt'], 'uyết': ['úyêt', 'uýêt'],
    'yệt': ['ỵêt'], 'yết': ['ýêt'],
    'iếu': ['íêu'], 'iều': ['ìêu'], 'iệu': ['ịêu'], 'iễu': ['ĩêu'], 'iểu': ['ỉêu'],
    'yếu': ['ýêu', 'yêú'], 'yểu': ['ỷêu', 'yêủ'],
    'uỵp': ['ụyp'], 'uýp': ['úyp'], 'uýt': ['úyt'], 'uỵt': ['ụyt'],
    'ìu': ['iù'], 'ỉu': ['iủ'], 'ĩu': ['iũ'], 'íu': ['iú'], 'ịu': ['iụ'],
    'uỵu': ['uyụ', 'ụyu'], 'uỷu': ['uyủ', 'ủyu'],
    'ỏi': ['oỉ'], 'ói': ['oí'], 'òi': ['oì'], 'ọi': ['oị'], 'õi': ['oĩ'],
    'oóc': ['óoc'], 'oòng': ['òong'], 'oóng': ['óong'],
    'ổi': ['ôỉ'], 'ối': ['ôí'], 'ồi': ['ôì'], 'ội': ['ôị'], 'ỗi': ['ôĩ'],
    'ởi': ['ơỉ'], 'ới': ['ơí'], 'ời': ['ơì'], 'ợi': ['ơị'], 'ỡi': ['ơĩ'],
    'ùa': ['uà'], 'úa': ['uá'], 'ủa': ['uả'], 'ũa': ['uã'], 'ụa': ['uạ'],
    'ùi': ['uì'], 'úi': ['uí'], 'ủi': ['uỉ'], 'ũi': ['uĩ'], 'ụi': ['uị'],
    'ùy': ['uỳ'], 'úy': ['uý'], 'ủy': ['uỷ'], 'ũy': ['uỹ'], 'ụy': ['uỵ'],
    'uốc': ['úôc'], 'uồc': ['ùôc'], 'uổc': ['ủôc'], 'uộc': ['ụôc'], 'uỗc': ['ũôc'],
    'uối': ['úôi', 'uôí'], 'uồi': ['ùôi', 'uôì'], 'uổi': ['ủôi', 'uôỉ'], 'uội': ['ụôi', 'uôị'], 'uỗi': ['ũôi', 'uôĩ'],
    'ối': ['ôí'], 'ồi': ['ôì'], 'ổi': ['ôỉ'], 'ội': ['ôị'], 'ỗi': ['ôĩ'],
    'uốn': ['úôn'], 'uồn': ['ùôn'], 'uổn': ['ủôn'], 'uộn': ['ụôn'], 'uỗn': ['ũôn'],
    'uống': ['úông'], 'uồng': ['ùông'], 'uổng': ['ủông'], 'uộng': ['ụông'], 'uỗng': ['ũông'],
    'uốt': ['úôt'], 'uột': ['ụôt'],
    'ữa': ['ưã'], 'ựa': ['ưạ'], 'ứa': ['ưá'], 'ừa': ['ưà'], 'ửa': ['ửa'],
    'ữi': ['ưĩ'], 'ựi': ['ưị'], 'ứi': ['ưí'], 'ừi': ['ưì'], 'ửi': ['ửi'],
    'ước': ['ứơc'], 'ườc': ['ừơc'], 'ược': ['ựơc'], 'ưỡc': ['ữơc'],
    'ưới': ['ứơi', 'ươí'], 'ười': ['ừơi', 'ươì'], 'ượi': ['ựơi', 'ươị'], 'ưỡi': ['ữơi', 'ươĩ'],
    'ướm': ['ứơm'], 'ườm': ['ừơm'], 'ượm': ['ựơm'], 'ưỡm': ['ữơm'],
    'ướn': ['ứơn'], 'ườn': ['ừơn'], 'ượn': ['ựơn'], 'ưỡn': ['ữơn'],
    'ướng': ['ứơng'], 'ường': ['ừơng'], 'ượng': ['ựơng'], 'ưỡng': ['ữơng'], 'ượng': ['ựơng'],
    'ướp': ['ứơp'], 'ườp': ['ừơp'], 'ượp': ['ựơp'], 'ưỡp': ['ữơp'],
    'ướt': ['ứơt'], 'ượt': ['ựơt'],
    'ướu': ['ứơu', 'ươú'], 'ườu': ['ừơu', 'ươù'], 'ượu': ['ựơu', 'ươụ'],
    'ựu': ['ưụ'], 'ừu': ['ưù'], 'ửu': ['ưủ'], 'ứu': ['ưú'], 'ữu': ['ưũ'],
    'oáng': ['óang'], 'oạch': ['ọach'], 'oách': ['óach'],
    'oán': ['óan'], 'oạn': ['ọan'], 'oãn': ['õan'],'oàn':['òan'],
    'oắn': ['óăn'],
    'oặt': ['ọăt'], 'uắc': ['úăc'],
    'uấy': ['úây', 'uâý'], 'uầy': ['ùây', 'uâỳ'], 'uẩy': ['ủây', 'uâỷ'], 'uậy': ['ụây', 'uâỵ'],
    'ưới': ['ứơi', 'ươí'], 'ười': ['ừơi', 'ươì'], 'ượi': ['ựơi', 'ươị'], 'ưỡi': ['ữơi', 'ươĩ'],
    'ưở':['ửơ'],
    'uèn': ['ùen'],
    'uyến': ['úyên', 'uýên'], 'uyền': ['ùyên', 'uỳên'], 'uyển': ['ủyên', 'uỷên'], 'uyện': ['ụyên', 'uỵên'], 'uyễn':['ũyên', 'uỹên'],
    'uầy': ['ùây'], 'uỷu': ['ủyu', 'uyủ'], 'uỳnh': ['ùynh'],
    'uốc': ['úôc'], 'uần': ['ùân'],
    'yến':['ýên'], 'yền': ['ỳên'], 'yển':'ỷên','yễn':['ỹên'], 'yện':['ỵên']
}


tonal_mapping_2 = {
    'ấy': ['âý'], 'ầy': ['âỳ'], 'ẩy': ['âỷ'], 'ậy': ['âỵ'],
    'ữa': ['ưã'], 'ựa': ['ưạ'], 'ứa': ['ưá'], 'ừa': ['ưà'], 'ửa': ['ửa'],
    'ướng': ['ứơng'], 'ường': ['ừơng'], 'ượng': ['ựơng'], 'uyết':['uýêt'],
    'uyến':['uýên'], 'uyền':['uỳên'], 'uyển':['uỷên']
}


class WrongWordGenerator:
    def __init__(self):
        self.__word = ""

    def get_word(self):
        return self.__word

    def set_word(self, word: str):
        self.__word = word
    
    @staticmethod
    def __swap(word, pos_1, pos_2) -> str:
        tmp = word[pos_1]
        word = word.replace(word[pos_1], word[pos_2], 1)
        word = word.replace(word[pos_2], tmp, 1)
        return word

    def remove_char(self) -> str:
        """
        Example:
            inp: "hello"
            out: "ello" or "hllo" or "helo" or"hell" or "hell"
        Note: This method is used for both en & vn
        """
        word = self.get_word()
        pos = random.randrange(0, len(word))
        
        # remove char
        word = word[: pos] + word[pos + 1:]
        return word

    def insert_char_english(self) -> str:
        """
        Example:
            inp: "hi"
            out: "ahi" or "bhi" or "chi" or ...
        """
        word = self.get_word()
        pos_1 = random.randrange(0, len(word))

        # select random char
        alphabet = string.ascii_lowercase  # Get all lowercase letters
        pos_2 = random.randrange(0, len(alphabet))
        
        # insert random char into word
        word = word[:pos_1] + alphabet[pos_2] + word[pos_1:]
        return word

    def swap_char_english(self) -> str:
        """
        Input: word

        Output: dict that key is the original word,
                value is the string of wrong words separated by delimiter ', '
        Example
            inp: hello
            out: {"hello": "its_permutations"}
        """
        word = self.get_word()
        pos_1 = random.randrange(0, len(word))
        pos_2 = random.randrange(0, len(word))
        
        # check for similar pos
        if len(word) > 2:
            while pos_2 == pos_1:
                pos_1 = random.randrange(0, len(word))
                pos_2 = random.randrange(0, len(word))
        return self.__swap(word, pos_1, pos_2)


    ##########################################################################
    def  wrong_tone_pos_vn(self) -> list:
        '''
        Examples: {"hiếu": "híêu, hiêú"}
                  {"giường": "giừơng"}
                  {quặng: ""}

        "gi" và "qu" are deemed to be consonants
        '''
        word = self.get_word()
        result = []
        # start code
        # start with "gi" hoặc "qu"
        if 'qu' in word or 'gi' in word:
            for key in tonal_mapping_2:
                if word.endswith(key):
                    for value in tonal_mapping_2[key]:
                        wrong_word = word[:-len(key)] + value
                        result.append(wrong_word)
        else:
            # Remaining cases
            for key in tonal_mapping_1 :
                if word.endswith(key):
                    for value in tonal_mapping_1 [key]:
                        wrong_word = word[:-len(key)] + value
                        result.append(wrong_word)
        return result


###############################################################################
def add_noise(sentence: str) -> str:
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
    wrong_word_generator = WrongWordGenerator()
    if len(sentence.split()) <= 10:
        max_wrong_words = 3
    elif 10 < len(sentence.split()) <= 20:
        max_wrong_words = 5
    elif len(sentence.split()) > 20:
        max_wrong_words = 7
    
    for i in range(max_wrong_words):
        # randomize index for word
        word_index = random.randrange(0, len(sentence.split()))

        wrong_word_generator.set_word(sentence.split()[word_index])

        # randomize method that's gonna be applied
        method_index = random.randrange(0, 3)
        if method_index == 0: # insert
            generated_result = wrong_word_generator.insert_char_english()

        elif method_index == 1: # remove
            generated_result = wrong_word_generator.remove_char()

        elif method_index == 2: # swap
            generated_result = wrong_word_generator.swap_char_english()
    return sentence.replace(sentence.split()[word_index], generated_result)


def add_noise_eng() -> None:
    num_of_wrong_sentence = 10

    with open('en_corpus.txt', 'r') as reader:
        counter = 0
        for line in reader:
            print(line)
            with open('En_noised_data/' + f'{counter}.txt', 'w') as writer:
                for i in range(num_of_wrong_sentence):
                    incorect_sentence = add_noise(line)
                    correct_sentence = line

                    # ensure incorrect sentence must differ from correct sentence
                    while incorect_sentence == correct_sentence:
                        incorect_sentence = add_noise(line)
                
                    writer.write(correct_sentence[:-1] + '|' + incorect_sentence)
            counter += 1
    return None


###############################################################################
def add_noised_vn() -> None:
    pass

###############################################################################
def main() -> None:
    add_noise_eng()
    # add_noised_vn()
    return None


if __name__ == '__main__':
    main()
    

