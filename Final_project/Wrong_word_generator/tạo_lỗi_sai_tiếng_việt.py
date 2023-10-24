"""
Reference:
    1/ https://vi.wikipedia.org/wiki/Quy_t%E1%BA%AFc_%C4%91%E1%BA%B7t_d%E1%BA%A5u_thanh_c%E1%BB%A7a_ch%E1%BB%AF_Qu%E1%BB%91c_ng%E1%BB%AF
    
    2/ https://gist.github.com/nguyenvanhieuvn/72ccf3ddf7d179b281fdae6c0b84942b?fbclid=IwAR2jH7XiK_nUeOO0muigSfHwIqkHq_pt7rE2UfO_vaP3AGhUGy03-a3sxq0

Telex rule: Telex requires the user to type in a base letter, followed by one or two characters that represent the diacritical marks:
"""
import string
import re


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


  def punctuation_marks_wrong_vn(self) -> dict:
      '''
      input: "hiếu"
      output:
        {"hiếu": ["híêu", "hiêú"]}
      Có 2 trường hợp đặc biệt là "gi" và "qu"
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

def test_punctuation_marks_wrong_vn() -> None:
  word = 'giường'
  generator = wrong_word_generator()
  generator.set_word(word)
  wrong_word = generator.punctuation_marks_wrong_vn()
  print(wrong_word)
  return None


def main() -> None:
    # test_telex_wrong_vn()
    test_punctuation_marks_wrong_vn()


if __name__ == '__main__':
  main()




"""Telex"""
vowel_ls = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
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

Tone_markings = ['', 'f', 's', 'r', 'x', 'j'] # level, falling, rising, dipping-rising, rising glottalized, falling glottalized

nguyen_am_to_ids = {}

uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"


def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|\
                ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|\
                Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
    
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|\
                ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|\
                Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
    
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic


dicchar = loaddicchar()



def convert_unicode(txt):
  return re.sub(r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',\
                lambda x: dicchar[x.group()], txt)





class wrong_word_generator:
  def __init__(self):
      self.__word = None
      self.nguyen_am_to_ids = {}
      for i in range(len(vowel_ls)):
          for j in range(len(vowel_ls[i]) - 1):
              self.nguyen_am_to_ids[vowel_ls[i][j]] = (i, j)
      print(self.nguyen_am_to_ids)


  def get_word(self):
      return self.__word


  def set_word(self, word: str):
      self.__word = word


  def vn_word_to_telex_type(self, word):
      dau_cau = 0
      new_word = ''
      # tượng
      # tuonwg jt
      # tuowjng ww
      # tuonwgj j
      # tuowjng n
      # tuwojng g

      # huojnwg w
      # huongw 
      for char in word:
          x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
          print(x,y, new_word)
          if x == -1:
              new_word += char
              continue
          if y != 0:
              dau_cau = y
          new_word += vowel_ls[x][-1]

      new_word += Tone_markings[dau_cau]
      return new_word


  def vn_sentence_to_telex_type(self, sentence):
      words = sentence.split()
      for index, word in enumerate(words):
          words[index] = self.vn_word_to_telex_type(word)
      return ' '.join(words)


  def telex_wrong_vn(self) -> dict:
    '''
    input: "thủy"
    output: {"thủy": "thuyr"}
    '''
    word = self.get_word()
    result ={word: self.vn_sentence_to_telex_type(word)}
    return result


def test_telex_wrong_vn() -> None:
    word = 'tượng'
    generator = wrong_word_generator()
    generator.set_word(word)
    wrong_word = generator.telex_wrong_vn()
    print(wrong_word)
    return None

    