"""
Structure:
wrong_word_generator
    # For both
    remove_char()

    # For En
    insert_char_english()
    swap_char_english()

    # For Vn
    wrong_pos_tone_vn()


Telex Reference:
    1/ https://vi.wikipedia.org/wiki/Quy_t%E1%BA%AFc_%C4%91%E1%BA%B7t_d%E1%BA%A5u_thanh_c%E1%BB%A7a_ch%E1%BB%AF_Qu%E1%BB%91c_ng%E1%BB%AF
    
    2/ https://gist.github.com/nguyenvanhieuvn/72ccf3ddf7d179b281fdae6c0b84942b?fbclid=IwAR2jH7XiK_nUeOO0muigSfHwIqkHq_pt7rE2UfO_vaP3AGhUGy03-a3sxq0

Telex rule: Telex requires the user to type in a base letter, followed by one or two characters that represent the diacritical marks:
"""
import re
import random
import string
from itertools import permutations


tonal_mapping = {
    'ảo':['aỏ'], 'áo': ['aó'], 'ào':['aò'], 'ạo':['aọ'], 'ão':['aõ'],
    'ải':['aỉ'], 'ái': ['aí'], 'ài':['aì'], 'ại':['aị'], 'ãi':['aĩ'],
    'ảu':['aủ'], 'áu': ['aú'], 'àu':['aù'], 'ạu':['aụ'], 'ãu':['aũ'],
    'ảy':['aỷ'], 'áy': ['aý'], 'ày':['aỳ'], 'ạy':['aỵ'], 'ãy':['aỹ'],
    'òa':['oà'], 'óa':['oá'], 'ỏa':['oả'], 'õa':['oã'], 'ọa':['oạ'],
    'oái':['óai', 'oaí'], 'oài':['òai', 'oaì'], 'oải':['ỏai', 'oaỉ'], 'oại':['ọai', 'oaị'],
    'oáy':['óay', 'oaý'], 'oạy':['ọay', 'oaỵ'],
    'oạc':['ọac'], 'oác':['óac'],
    'oắc':['óăc'],'oặc':['ọăc'],
    'oằm':['òăm'],
    'oắn':['óăn'],'oằn':['òăn'],'oắn':['óăn'],'oẳn':['ỏăn'],
    'oằng':['òăng'], 'oẵng': ['õăng'], 'oắng':['óăng'],
    'oắt':['óăt'], 'oặt':['ọăt'],
    'uần':['ùân'], 'uẩn':['ủân'], 'uẫn':['ũân'], 'uận':['ụân'],
    'uặng':['ụặng'], 'uầng':['ùâng'], 'uẫng':['ũâng'], 'uẩng':['ủâng'],
    'uật':['ụât'], 'uất':['úât'],
    'ẩu':['âủ'], 'ấu': ['âú'], 'ầu':['âù'], 'ậu':['âụ'], 'ẫu':['âũ'],
    'ẩy':['âỷ'], 'ấy': ['âý'], 'ầy':['âỳ'], 'ậy':['âỵ'], 'ẫy':['âỹ'],
    'uẩy':['uâỷ','ủây'], 'uấy':['uâý','úây'],   'uẫy':['uâỹ','ũây'],
    'òe':['oè'], 'óe':['oé'], 'ỏe':['oẻ'], 'õe':['oẽ'], 'ọe':['oẹ'],
    'oèn':['òen'], 'oẻn':['ỏen'],
    'oèo':['òeo','oeò'], 'oẹo':['ọeo','oẹo'], 'oẻo':['ỏeo','oeỏ'],
    'éo':['eó'], 'èo':['eò'], 'ẹo':['eọ'], 'ẻo':['eỏ'], 'ẽo':['eõ'],
    'oét':['óet'], 'oẹt':['ọet'],
    'uệ':['ụê'], 'uế':['úê'],'uễ':['ũê'], 'uề':['ùê'], 'uể':['ủê'],
    'uệch':['ụêch'],
    'uềnh':['ùênh'], 'uếnh':['úênh'],'uệnh':['ụênh'],
    'ều':['êù'], 'ếu':['êú'], 'ệu':['êụ'], 'ễu':["êũ"], 'ểu':['êủ'],
    'ùy':['uỳ'], 'úy':['uý'], 'ủy':['uỷ'], 'uỵ':['ụy'],
    'ìa': ['ià'],'ĩa':['iã'], 'ỉa':['iả'], 'ía':['iá'], 'ịa':['iạ'],
    'uỵch':['ụych'], 'uých': ['úych'],
    'iếc':['íêc'], 'iệc':['ịêc'],
    'iềm':['ìêm'],'iểm':['ỉêm'], 'iếm':['íêm'], 'iệm':['ịêm'],
    'yểm':['ỷêm'],'yến':['ýên'],
    'iền':['ìên'],'iến':['íên'], 'iễn':['ĩên'],'iển':['ỉên'],'iện':['ịên'],
    'uyến':['úyên', 'uýên'], 'uyền':['ùyê', 'uỳên'], 'uyển':['ủyên', 'uỷên'], 'uyện':['ụyên', 'uỵên'], 'uyễn':['ũyên', 'uỹên'],
    'iếu':['íêu','iêú'], 'iều':['ìêu', 'iêù'], 'iệu':['ịêu', 'iêụ'],'iểu':['ỉêu', ' iêủ'],
    'iếng':['íêng'], 'iềng':['ìêng'], 'iệng':['ịêng'],'iểng':['ỉêng'],
    'iếp':['íêp'], 'iềp':['ìêp'], 'iệp':['ịêp'],
    'iết':['íêt'], 'iềt':['ìêt'], 'iệt':['ịêt'],
    'uyệt':['ụyêt','uỵêt'], 'uyết':['úyêt', 'uyết'],
    'iếu':['íêu'], 'iều':['ìêu'], 'iệu':['ịêu'],'iễu':['ĩêu'],'iểu':['ỉêu'],
    'yếu':['ýêu', 'yêú'], 'yểu':['ỷêu','yêủ'],
    'uỵp':['ụyp'], 'uýp':['úyp'], 'uýt':['úyt'], 'uỵt':['ụyt'],
    'ìu':['iù'], 'ỉu':['iủ'], 'ĩu':['iũ'], 'íu':['iú'], 'ịu':['iụ'],
    'uỵu':['uyụ', 'ụyu'], 'uỷu':['uyủ', 'ủyu'],
    'ỏi':['oỉ'], 'ói': ['oí'], 'òi':['oì'], 'ọi':['oị'], 'õi':['oĩ'],
    'oóc':['óoc'], 'oòng':['òong'], 'oóng':['óong'],
    'ổi':['ôỉ'], 'ối': ['ôí'], 'ồi':['ôì'], 'ội':['ôị'], 'ỗi':['ôĩ'],
    'ởi':['ơỉ'], 'ới': ['ơí'], 'ời':['ơì'], 'ợi':['ơị'], 'ỡi':['ơĩ'],
    'ùa':['uà'], 'úa':['uá'], 'ủa':['uả'], 'ũa':['uã'], 'ụa':['uạ'],
    'ùi':['uì'], 'úi':['uí'], 'ủi':['uỉ'], 'ũi':['uĩ'], 'ụi':['uị'],
    'uốc':['úôc'], 'uồc':['ùôc'], 'uổc':['ủôc'], 'uộc':['ụôc'], 'uỗc':['ũôc'],
    'uối':['úôi', 'uôí'], 'uồi':['ùôi', 'uôì'], 'uổi':['ủôi', 'uôỉ'], 'uội':['ụôi', 'uôị'], 'uỗi':['ũôi', 'uôĩ'],
    'uốn':['úôn'], 'uồn':['ùôn'], 'uổn':['ủôn'], 'uộn':['ụôn'], 'uỗn':['ũôn'],
    'uống':['úông'], 'uồng':['ùông'], 'uổng':['ủông'], 'uộng':['ụông'], 'uỗng':['ũông'],
    'uốt':['úôt'], 'uột':['ụôt'],
    'ữa':['ưã'], 'ựa':['ưạ'], 'ứa':['ưá'], 'ừa':['ưà'], 'ửa':['ửa'],
    'ữi':['ưĩ'], 'ựi':['ưị'], 'ứi':['ưí'], 'ừi':['ưì'], 'ửi':['ửi'],
    'ước':['ứơc'], 'ườc':['ừơc'], 'ược':['ựơc'], 'ưỡc':['ữơc'],
    'ưới':['ứơi', 'ươí'], 'ười':['ừơi', 'ươì'], 'ượi':['ựơi', 'ươị'], 'ưỡi':['ữơi', 'ươĩ'],
    'ướm':['ứơm'], 'ườm':['ừơm'], 'ượm':['ựơm'], 'ưỡm':['ữơm'],
    'ướn':['ứơn'], 'ườn':['ừơn'], 'ượn':['ựơn'], 'ưỡn':['ữơn'],
    'ướng':['ứơng'], 'ường':['ừơng'], 'ượng':['ựơng'], 'ưỡng':['ữơng'],'ượng':['ựơng'],
    'ướp':['ứơp'], 'ườp':['ừơp'], 'ượp':['ựơp'], 'ưỡp':['ữơp'],
    'ướt':['ứơt'], 'ượt':['ựơt'],
    'ướu':['ứơu', 'ươú'], 'ườu':['ừơu', 'ươù'],'ượu':['ựơu', 'ươụ'],
    'ựu':['ưụ'], 'ừu':['ưù'], 'ửu':['ưủ'], 'ứu':['ưú'], 'ữu':['ưũ'],
    'oáng':['óang'], 'oạch':['ọach'],'oách':['óach'],
    'oắn':['óăn'],
    'oặt':['ọăt'], 'uắc': ['úăc'],
    'uấy':['úây', 'uâý'], 'uầy':['ùây', 'uâỳ'], 'uẩy':['ủây', 'uâỷ'], 'uậy':['ụây', 'uâỵ'],
    'ưới':['ứơi', 'ươí'], 'ười':['ừơi', 'ươì'], 'ượi':['ựơi', 'ươị'], 'ưỡi':['ữơi', 'ươĩ'],
    'uèn':['ùen'],
    'uyến':['úyên', 'uýên'], 'uyền':['ùyên', 'uỳên'], 'uyển':['ủyên', 'uỷên'], 'uyện':['ụyên', 'uỵên'], 'uyễn':['ũyên', 'uỹên'],
    'uầy':['ùây'], 'uỷu':['ủyu','uyủ'], 'uỳnh':['ùynh'],
    'uốc':['úôc'], 'uần':['ùân']
}


class wrong_word_generator:
    def __init__(self):
        self.__word = None


    def get_word(self):
        return self.__word


    def set_word(self, word: str):
        self.__word = word


    def remove_char(self) -> dict:
        '''
        Input
            word: Input word to process

        Output
            result: dict that key is the original word,
                    value is the string of wrong words separated by delimiter ', '
        Example:
            inp: "hello"
            out: {"hello": "ello, hllo, helo, hell, hell"}
            
            in: "nguyễn"
            out: {'nguyễn': 'guyễn, nuyễn, ngyễn, nguễn, nguyn, nguyễ'}
        
        Note: This method is used for both en & vn
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
        word = self.get_word()
        swap_permutations = [''.join(p) for p in permutations(word) if p != word]
        return {word: ', '.join(swap_permutations)}


    def wrong_pos_tone_vn(self) -> dict:
        '''
        Input:
            single word

        Output:
            dict of string vals

        Examples: {"hiếu": "híêu, hiêú"}
                  {"giường": "giừơng"}
                  {quặng: ""}

        "gi" và "qu" are deemed to be consonants
        '''
        word = self.get_word()
        result ={word:[]}
        # start code
        # trường hợp có "gi" hoặc "qu" thì return word
        if 'qu' in word or 'gi' in word:
          return {word: [word]}

        else:
        # trường hợp thường
            for key in tonal_mapping:
                if word.endswith(key):
                    for value in tonal_mapping[key]:
                        wrong_word = word[:-len(key)] + value
                        result[word].append(wrong_word)
            if not result[word]:
                return {word: [word]}
            else:
                return result


###############################################################################
def test_remove_char_english() -> None:
    word = 'hello'
    generator = wrong_word_generator()
    generator.set_word(word)
    wrong_word = generator.remove_char()
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


###############################################################################
def test_remove_char_vn() -> None:
    word = 'Nguyễn'
    generator = wrong_word_generator()
    generator.set_word(word)
    wrong_word = generator.remove_char()
    print(wrong_word)
    return None

def test_wrong_pos_tone_vn() -> None:
    word_ls  = ['nguyễn', 'giường']
    for word in word_ls:
        generator = wrong_word_generator()
        generator.set_word(word)
        wrong_word = generator.wrong_pos_tone_vn()
        print(wrong_word)
    return None

################################################################################
def main() -> None:
    # test_remove_char_english()    
    # test_insert_char_english()
    # test_swap_char_english()

    # test_remove_char_vn()
    test_wrong_pos_tone_vn()
    return None


if __name__ == '__main__':
    main()
