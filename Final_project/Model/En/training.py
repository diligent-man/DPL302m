import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import zipfile




def main() -> None:
    with zipfile('../../Data_preping/Data_gathering/En/Preprocessed_data/noised_corpus.zip', 'r') as z:
        z.extractall()
        print('Dataset extracted')

    return None


if __name__ == '__main__':
    main()