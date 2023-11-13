from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


def horizontal_bar_plot(x, y, x_label, y_label, title):
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')
    fig.canvas.manager.set_window_title('Eldorado K-8 Fitness Chart')

    ax.barh(y, x, align='center')
    # ax.set_yticks(y_pos, labels=people)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return ax



def main() -> None:
    metadata = {"category_ls_1": ['Business', 'Entertainment', 'Health', 'Sport', 'Style', 'Tech', 'Travel', 'Weather', 'World'],
                # "category_ls_2": ['Business', 'Entertainment', 'Health', 'Sport', 'Style', 'Travel', 'Weather'],
                "category_ls_3": ['Ted_talk/'],
                
                "data_dir_1": ['../Preprocessed_data/BBC/'],
                # "data_dir_2": ['../Data/GPT'],
                "data_dir_3": ['../Preprocessed_data/']
                }

    index_1 = index_2 = index_3 = 0
    sentence_len = []  # dict of words for visualizing word freq
    word_freq = {}  # dict of words for visualizing word freq
    for i in range(len(metadata["category_ls_1"]) + len(metadata["category_ls_3"])):
        # choosing what category will be processed
        if i < len(metadata["category_ls_1"]):
            flag = "1"
            category = metadata["category_ls_" + flag][index_1]
            index_1 += 1

        # elif i < len(metadata["category_ls_1"]) + len(metadata["category_ls_2"]):
        #     flag = "2"
        #     category = metadata["category_ls_" + flag][index_2]
        #     index_2 += 1

        elif i < len(metadata["category_ls_1"]) + len(metadata["category_ls_3"]):
            flag = "3"
            category = metadata["category_ls_" + flag][index_3]
            index_3 += 1
        path = metadata["data_dir_" + flag][0] + category + "/"
        source = PlaintextCorpusReader(path, '.*txt')

        # compute sentence len
        sentences = source.sents()
        for sentence in sentences:
            sentence_len.append(len(sentence))

        # compute word freq
        words = source.words()
        words = [WordNetLemmatizer().lemmatize(word) for word in words]
        for word in words:
            if word not in word_freq.keys():
                word_freq[word] = 1
            else:
                word_freq[word] += 1

    # visualize dist of sentence len
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')
    ax.hist(sentence_len, bins=100)
    ax.set_xlabel("Sentence length")
    ax.set_ylabel("Freq")
    ax.set_title("Distribution of sentence length in dataset")
    plt.show()


    # word_freq
    word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
    # take top 10
    word_ls = list(word_freq.keys())[:10]
    freq_ls = [word_freq[i] for i in word_ls]
    horizontal_bar_plot(x=freq_ls, y=word_ls, x_label="Freq", y_label="Word", title="Top 10 highest word freq in dataset")
    plt.show()
    






    return None


if __name__ == '__main__':
    main()