import os
import shutil

def main() -> None:
    if 'corpus.txt' in os.listdir('../Preprocessed_data/'):
            os.remove(path='../Preprocessed_data' + '/' + 'corpus.txt')

    metadata = {"category_ls_1": ['Business', 'Entertainment', 'Health', 'Sport', 'Style', 'Tech', 'Travel', 'Weather','World'],
                "category_ls_2": ['Ted_talk', 'Wiki'],
                "data_dir_1": ['../Data/BBC'],
                "data_dir_2": ['../Data'],
                "preprocessed_data_dir_1": ["../Preprocessed_data/BBC"],
                "preprocessed_data_dir_2": ["../Preprocessed_data"]
                     }
    index_1 = index_2 = 0
    for i in range(len(metadata["category_ls_1"]) + len(metadata["category_ls_2"])):
        if i < len(metadata["category_ls_1"]):
            flag = "1"
            category = metadata["category_ls_" + flag][index_1]
            index_1 += 1
        else:
            flag = "2"
            category = metadata["category_ls_" + flag][index_2]
            index_2 += 1
        data_dir = metadata["preprocessed_data_dir_" + flag][0]
        data_dir = data_dir + '/' + category

        # read & write into synthesized.txt
        for i in range(len(os.listdir(data_dir))):
            print(data_dir, i)
            file = data_dir + '/' + os.listdir(data_dir)[i]
            with open(file=file, mode='r') as reader:
                with open (file='../Preprocessed_data/corpus.txt', mode='a') as writer:
                    writer.write(reader.read())
            


    return None


if __name__ == '__main__':
	main()