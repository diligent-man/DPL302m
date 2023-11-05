import os
import shutil


def main() -> None:
    if 'corpus.txt' in os.listdir('../Preprocessed_data/'):
            os.remove(path='../Preprocessed_data' + '/' + 'corpus.txt')

    category_ls = ['am_thuc', 'doi_song', 'du_lich', 'gia_dinh', 'the_gioi',
                   'giai_tri', 'giao_duc', 'khong_gian_song', 'loi_song',
                   'the_thao', 'thoi_su', 'thoi_trang']
    data_dir = '../Data'

    for category in category_ls:
        data_dir = '../Preprocessed_data' + '/' + category
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