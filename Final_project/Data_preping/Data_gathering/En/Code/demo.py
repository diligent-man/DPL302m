regex_dict = {0: [r"[~!@#$%\^&\*()\_,./<>\?;:\"\[\]\{\}\\\|“”\u2122\u00A90-9]*", ''],  # punctuation_marks_and_numeral
                     1: [r"[-–]", " "], # hyphen & dash
                     2: [r"[\u4E00 to \u9FFF]", " "] # Chinese hieroglyphs
                     }
for i in regex_dict:
    print(type(regex_dict[i][0]))
