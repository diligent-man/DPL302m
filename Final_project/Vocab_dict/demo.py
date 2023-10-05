import pandas as pd


def main() -> None:
    words = []
    words_alpha = []

    with open('english-wordlist/words.txt') as f:
        for word in f:
            words.append(word[:-1])
    words = pd.DataFrame({"word": words})

    with open('english-wordlist/words_alpha.txt') as f:
        for word in f:
            words_alpha.append(word[:-1])
    words_alpha = pd.DataFrame({"word": words_alpha})

    difference = [row for row in words.values if row not in words_alpha.values]
    print(difference)






    return None


if __name__ == '__main__':
    main()



