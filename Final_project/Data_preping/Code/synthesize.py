import os
import shutil
from sklearn.model_selection import train_test_split


def main() -> None:
    if "english_1.txt" in os.listdir("../../../../Modelling/En"):
        os.remove("../../../../Modelling/En/data/english_1.txt")

    metadata = {"category_ls_1": ['Business', 'Entertainment', 'Health', 'Sport', 'Style', 'Tech', 'Travel', 'Weather', 'World'],
                # "category_ls_2": ['Business', 'Entertainment', 'Health', 'Sport', 'Style', 'Travel', 'Weather'],
                "category_ls_3": ['Ted_talk'],
                "data_dir_1": ['../Data/BBC'],
                # "data_dir_2": ['../Data/GPT'],
                "data_dir_3": ['../Data'],
                "preprocessed_data_dir_1": ["../Preprocessed_data/BBC"],
                # "preprocessed_data_dir_2": ["../Preprocessed_data/GPT"],
                "preprocessed_data_dir_3": ["../Preprocessed_data"]
                }

    # split into singular sentence
    index_1 = index_2 = index_3 = 0
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

        data_dir = metadata["preprocessed_data_dir_" + flag][0]
        data_dir = data_dir + '/' + category

        # read data
        data = []
        for i in range(len(os.listdir(data_dir))):
            print(data_dir, i)
            file = data_dir + '/' + os.listdir(data_dir)[i]
            with open(file=file, mode='r') as f:
                for line in f:
                    data.append(line)

        train, test = train_test_split(data, test_size=0.1, random_state=12345, shuffle=True)

        # write to file
        with open(file='../../../../Modelling/En/data/english_1.txt', mode='a') as f:
            f.writelines(train)

    shutil.rmtree('../Preprocessed_data')
    return None


if __name__ == '__main__':
	main()