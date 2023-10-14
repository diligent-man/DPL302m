

import scipy.misc
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models, initializers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions

from tensorflow.python.framework.ops import EagerTensor
from resnets_utils import load_dataset, predict, forward_propagation_for_predict, OHE, rand_mini_batches





def main() -> None:
    Exercise_1()
    return None

if __name__ == '__main__':
    main()

