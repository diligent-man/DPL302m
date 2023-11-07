"""
Structure:
wrong_word_generator
    # For both
    remove_char()

    # For En
    insert_char()
    swap_char_english()

    # For Vn
     wrong_tone_pos_vn()


Telex Reference:
    1/ https://vi.wikipedia.org/wiki/Quy_t%E1%BA%AFc_%C4%91%E1%BA%B7t_d%E1%BA%A5u_thanh_c%E1%BB%A7a_ch%E1%BB%AF_Qu%E1%BB%91c_ng%E1%BB%AF
    
    2/ https://gist.github.com/nguyenvanhieuvn/72ccf3ddf7d179b281fdae6c0b84942b?fbclid=IwAR2jH7XiK_nUeOO0muigSfHwIqkHq_pt7rE2UfO_vaP3AGhUGy03-a3sxq0

Telex rule: Telex requires the user to type in a base letter, followed by one or two characters that represent the diacritical marks:
"""
from math import ceil
import os
import re
import string
import random
import numpy as np


#######################################################################################
# Two dicts for wrong_tone_pos funs
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
######################################################################################
# This section for expand_telex_error
bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']


######################################################################################
class WrongWordGenerator:
    def __init__(self):
        self.__word = ""

    def get_word(self):
        return self.__word

    def set_word(self, word: str):
        self.__word = word

    @staticmethod
    def __expand_telex_error(word) -> str:
        """

        - Insert cuối - trường hợp đặc biệt: 'thủyy -> code chạy: 'thuyyr' -> ko đúng. Đúng: 'thuyry'
        - Insert bất kỳ - chạy bình thường
        """
        nguyen_am_to_ids = {}
        for i in range(len(bang_nguyen_am)):
            for j in range(len(bang_nguyen_am[i]) - 1):
                nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)
        dau_cau = 0
        new_word = ''
        for char in word:
            x, y = nguyen_am_to_ids.get(char, (-1, -1))
            if x == -1:
                new_word += char
                continue
            if y != 0:
                dau_cau = y
            new_word += bang_nguyen_am[x][-1]
        new_word += bang_ky_tu_dau[dau_cau]

        # swap last char vs penultimate char
        # When insert into the last pos, it yields 'thủyy' -> 'thuyyr'
        # but it should be 'thuyry'
        pos_1 = len(new_word) - 1
        pos_2 = len(new_word) - 2
        
        tmp = new_word[pos_1]
        new_word = new_word.replace(new_word[pos_1], new_word[pos_2], 1)
        new_word = new_word.replace(new_word[pos_2], tmp, 1)
        return new_word

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

    def insert_char(self, language) -> str:
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

        bang_nguyen_am_co_dau = np.array(bang_nguyen_am)[:, 1:-1]
        bang_nguyen_am_co_dau = [ele for row in bang_nguyen_am_co_dau for ele in row]
        flag = True # check whether tonal vowel exist in word or not
        for char in bang_nguyen_am_co_dau:
            if char in word:
                flag = True
                break

        if language == 'vn' and flag == True:
            word = self.__expand_telex_error(word)
        return word

    def swap_char(self, language) -> str:
        word = self.get_word()

        pos_1 = random.randrange(0, len(word))
        pos_2 = random.randrange(0, len(word))
        
        # check for similar pos
        if len(word) > 2:
            while pos_2 == pos_1:
                pos_1 = random.randrange(0, len(word))
                pos_2 = random.randrange(0, len(word))
        # swap char
        tmp = word[pos_1]
        word = word.replace(word[pos_1], word[pos_2], 1)
        word = word.replace(word[pos_2], tmp, 1)

        # if vn, perform expand telex typo
        if language == 'vn':
            word = self.__expand_telex_error(word)
        return word


    ##########################################################################
    def wrong_tone_pos_vn(self) -> str:
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
        if len(result) == 0:
            return word
        if len(result) == 1:
            return result[0]
        elif len(result) > 1:
            word_index = random.randrange(0, len(result))
            return result[word_index]


###############################################################################
def add_noise(sentence: str, language: str) -> str:
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
        20 < len(sen) > 30: 7 wrong words
        10 < len(sen) > 20: 5 wrong words
        len(sen) <= 10: 3 wrong words
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
        if language == 'en':
            # randomize method that's gonna be applied
            method_index = random.randrange(0, 3)

            if method_index == 0: # remove
                generated_result = wrong_word_generator.remove_char()

            elif method_index == 1: # insert
                generated_result = wrong_word_generator.insert_char(language)

            elif method_index == 2: # swap
                generated_result = wrong_word_generator.swap_char(language)

        elif language == 'vn':
            # randomize method that's gonna be applied
            method_index = random.randrange(0, 4)

            if method_index == 0: # remove
                generated_result = wrong_word_generator.remove_char()

            elif method_index == 1: # insert
                generated_result = wrong_word_generator.insert_char(language)

            elif method_index == 2: # swap
                generated_result = wrong_word_generator.swap_char(language)
            
            elif method_index == 3: # wrong_tone_pos
                generated_result = wrong_word_generator.wrong_tone_pos_vn()
    return sentence.replace(sentence.split()[word_index], generated_result)


def generate_wrong_word(language) -> None:
    if language == "en":
        filename = 'en_corpus.txt'
        save_path = 'noised_en/'

    elif language == 'vn':
        filename = 'vn_corpus.txt'
        save_path = 'noised_vn/'

    # Calculate how many file will be created
    num_of_wrong_sentence = 10
    num_of_line_per_file = 100_000
    with open(filename, 'r') as f:
        num_of_sentence_in_corpus = len(f.readlines())
    ender_ls = [i * num_of_line_per_file for i in range(1, ceil(num_of_sentence_in_corpus * num_of_wrong_sentence / num_of_line_per_file)+2)]
    print(ender_ls)


    counter = 0
    starter = 0
    with open(filename, 'r') as reader:
        for i in range(len(ender_ls)):
            ender = ender_ls[i]
            filename = save_path + str(starter) + "_" + str(ender) + ".txt"

            # check ender whether it's over # of lines in corpus or not
            if ender > num_of_sentence_in_corpus * num_of_wrong_sentence:
                ender = num_of_sentence_in_corpus * num_of_wrong_sentence

            while counter < ender:
                line = reader.readline()
                for j in range(num_of_wrong_sentence):
                    incorect_sentence = add_noise(line, language)
                    correct_sentence = line

                    # ensure incorrect sentence must differ from correct sentence
                    while incorect_sentence == correct_sentence:
                        incorect_sentence = add_noise(line, language)

                    # write to file
                    with open(filename, 'a') as writer:
                        writer.write(correct_sentence[:-1] + '|' + incorect_sentence)

                    print(language, f'{counter} lines added noise')
                    counter += 1

                    # write into next file for the residual
                    if counter >= ender:
                        filename = save_path + str(ender) + "_" + str(ender_ls[i+1]) + ".txt"
                        for k in range(num_of_wrong_sentence-j-1):
                            incorect_sentence = add_noise(line, language)
                            correct_sentence = line

                            # ensure incorrect sentence must differ from correct sentence
                            while incorect_sentence == correct_sentence:
                                incorect_sentence = add_noise(line, language)

                            with open(filename, 'a') as writer:
                                print(correct_sentence[:-1] + '|' + incorect_sentence)
                                writer.write(correct_sentence[:-1] + '|' + incorect_sentence)

                            print(language, f'{counter} lines added noise')
                            counter += 1
                        break
            # update starter
            starter = ender
    return None


###############################################################################
def main() -> None:
    # generate_wrong_word("en")
    # generate_wrong_word("vn")
    return None


if __name__ == '__main__':
    main()