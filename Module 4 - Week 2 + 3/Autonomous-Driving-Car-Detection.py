import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image
#################################################################################################################################################################

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.6):

    box_scores = box_confidence * box_class_probs # tính score của mỗi hộp dự đoán
    box_classes = tf.argmax(box_scores, axis = -1) #có tác dụng tìm ra class có điểm số cao nhất cho mỗi hộp trong mỗi ô.
    box_class_scores = tf.reduce_max(box_scores, axis = -1, keepdims=False) # tìm sorce cao nhất cho từng hộp trong mỗi ô.
    filtering_mask = (box_class_scores >= threshold)
    #Áp dụng mask cho box_class_scores, hộp và box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    return scores, boxes, classes

##IOU
def iou(box1, box2):

    (box1_x1, box1_y1, box1_x2, box1_y2) = box1 #(x1,y1) góc trái dưới / (x2,y2) góc phải trên
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    # Nếu không có phần giao nhau giữa box1 và box2, thì các giá trị này sẽ là 0.
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(0, yi2 - yi1)
    inter_height = max(0, xi2 - xi1)
    inter_area = inter_width * inter_height
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_x2 - box1_x1) * ((box1_y2 - box1_y1))
    box2_area = (box2_x2 - box2_x1) * ((box2_y2 - box2_y1))
    union_area = box1_area + box2_area - inter_area
    # compute the IoU
    iou = inter_area / union_area
    return iou

##Non-max Suppression
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    #stores the maximum number of cartridges that you want to keep behind the NMS.
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    #Determine which bounding boxes should be kept
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)
    scores = tf.gather(scores,nms_indices)
    boxes = tf.gather(boxes,nms_indices)
    classes = tf.gather(classes,nms_indices)
    return scores, boxes, classes

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, score_threshold)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    return scores, boxes, classes

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
model_image_size = (608, 608) # Same as yolo_model input layer size
##Loading a Pre-trained Model
yolo_model = load_model("model_data/", compile=False)
yolo_model.summary()

def predict(image_file):

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1], image.size[0]], 10, 0.3, 0.5)

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), "images/" + image_file))
    # Generate colors for drawing bounding boxes.
    colors = get_colors_for_classes(len(class_names))
    # Draw bounding boxes on the image file
    # draw_boxes2(image, out_scores, out_boxes, out_classes, class_names, colors, image_shape)
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=100)
    # Display the results in the notebook
    output_image = Image.open(os.path.join("out", image_file))
    imshow(output_image)
    return out_scores, out_boxes, out_classes

out_scores, out_boxes, out_classes = predict("test.jpg")

