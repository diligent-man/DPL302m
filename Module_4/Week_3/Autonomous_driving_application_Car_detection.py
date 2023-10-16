import os
import argparse
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as kr
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Inp
        boxes -- tensor of shape (19, 19, 5, 4)
        box_confidence -- tensor of shape (19, 19, 5, 1)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Out
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    ### START CODE HERE
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    # IMPORTANT: set axis to -1
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold

    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    return scores, boxes, classes

def Exercise_1() -> None:
    tf.random.set_seed(10)
    # Init
    box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)

    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.5)
    print("scores[2] = " + str(scores[2].numpy()))
    print("boxes[2] = " + str(boxes[2].numpy()))
    print("classes[2] = " + str(classes[2].numpy()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))

    assert scores.shape == (1789,), "Wrong shape in scores"
    assert boxes.shape == (1789, 4), "Wrong shape in boxes"
    assert classes.shape == (1789,), "Wrong shape in classes"

    assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
    assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
    assert classes[2].numpy() == 8, "Values are wrong on classes"

    print("\033[92m All tests passed!")
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
    Exercise_1()
    return None


if __name__ == '__main__':
    main()
