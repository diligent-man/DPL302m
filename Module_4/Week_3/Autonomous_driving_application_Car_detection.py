import os
import argparse
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as kr
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow



def test_pretrained_YOLO() -> None:
    class_names = read_classes("model_data/coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    model_image_size = (608, 608) # Same as yolo_model input layer size


    return None
def main() -> None:
    """
                                            Inputs and outputs of YOLO
    Input: batch of imgs has shape (608, 608, 3)

    Output: (pc, bx, by, bh, bw, c) where
        pc: confidence score
        bx: bounding box's x centroid
        by: bounding box's y centroid
        bh: bounding box's height
        bw: bounding box's width
        c: belonging class
        -> # of class <==> # of components in bounding box

                                                    Anchor boxes
        Chosen by exploring the training data to choose reasonable height/width ratios
        that represent the different classes

        Dim of the encoding tensors from 2nd to last dim based on anchor boxes is (m, nh, nw, anchors, classes)
        e.g inp (m, 608, 608, 3) -> deep cnn -> feature map (m, 19, 19, 5, ,85)



    """
        test_pretrained_YOLO()
    return None


if __name__ == '__main__':
    main()
