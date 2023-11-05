import os
import shutil


def main() -> None:
    if 'vn_corpus.txt' in os.listdir('../../../../Wrong_word_generator/'):
            os.remove(path='../../../../Wrong_word_generator/' + 'vn_corpus.txt')

    category_ls = ['am_thuc', 'doi_song', 'du_lich', 'gia_dinh', 'the_gioi',
                   'giai_tri', 'giao_duc', 'khong_gian_song', 'loi_song',
                   'the_thao', 'thoi_su', 'thoi_trang']
    data_dir = '../Preprocessed_data/'

    for category in category_ls:
        # read & write into vn_corpus.txt
        for i in range(len(os.listdir(data_dir + category))):
            file = data_dir + category + "/" + os.listdir(data_dir + category)[i]
            print(file)
            with open(file=file, mode='r') as reader:
                with open (file='../../../../Wrong_word_generator/vn_corpus.txt', mode='a') as writer:
                    writer.write(reader.read())
    shutil.rmtree('../Preprocessed_data/')
    return None


if __name__ == '__main__':
	main()