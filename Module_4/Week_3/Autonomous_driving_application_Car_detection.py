import os
import cv2 as cv
import numpy as np
import tensorflow as tf

from tensorflow import keras as kr
from PIL import Image
from yad2k.keras_yolo import yolo_head
from yad2k.utils import scale_boxes, preprocess_image, read_anchors, read_classes, draw_boxes
from yad2k.draw_boxes import get_colors_for_classes




def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Inp
        boxes: (19, 19, 5, 4) <==> (cell, cell, anchors, anchors_coord)
        box_confidence: (19, 19, 5, 1) <==> (cell, cell, anchors, p_c)
        box_class_probs: (19, 19, 5, 80) <==> (cell, cell, anchors, c_i)
        threshold: hyperpara for discarding boxes

    Out
        scores: class prob score for selected boxes
            -> (None, )
        boxes: selected boxes
            -> (None, anchors_coord)
        classes: index of the detected class of selected boxes
            -> (None, )

    Note: "None" is here because you don't know the exact number of selected boxes,
          as it depends on the threshold.
          For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    # Step 1: Compute box scores
    # (19, 19, 5, 80) = (19, 19, 5, 1) * (19, 19, 5, 80)
    confidence_scores = box_confidence * box_class_probs # p_c * c_i
    # print(confidence_scores[0,0,0,20], confidence_scores[0,0,1,74], confidence_scores[0,0,2,8], confidence_scores[0,0,3,67], confidence_scores[0,0,4,56])

    # Step 2: Find index of max confidence score boxex & its vals
    # (19,19,5)
    max_score_boxes = tf.math.argmax(confidence_scores, axis=-1) # index of boxes having max conf score in 19*19 grid cell
    # (19,19,5)
    max_score_boxes_val = tf.math.reduce_max(confidence_scores, axis=-1) # their vals
    # print(max_score_boxes[0,0,:])
    # print(max_score_boxes_val[0,0,:])

    # Step 3: Create mask based on max_score_boxes_val by using "threshold"
    mask = max_score_boxes_val >= threshold

    # Step 4: Apply mask to max_score_boxes, boxes & max_score_boxes_val to retrieve satisfied boxes
    scores = tf.boolean_mask(tensor=max_score_boxes_val, mask=mask) # (None, )
    boxes = tf.boolean_mask(tensor=boxes, mask=mask) # (None, 4)
    classes = tf.boolean_mask(tensor=max_score_boxes, mask=mask) # (None, )
    return scores, boxes, classes

def Exercise_1() -> None:
    tf.random.set_seed(10)
    # Init
    box_confidence = tf.random.normal(shape=[19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random.normal(shape=[19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random.normal(shape=[19, 19, 5, 80], mean=1, stddev=4, seed = 1)

    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.5)

    assert scores.shape == (1789,), "Wrong shape in scores"
    assert boxes.shape == (1789, 4), "Wrong shape in boxes"
    assert classes.shape == (1789,), "Wrong shape in classes"
    assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
    assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
    assert classes[2].numpy() == 8, "Values are wrong on classes"
    print("\033[92m All tests passed!")
    return None


#################################################################################################################
def iou(box1, box2):
    """
    Inp:
    box1: (x_1, y_1, x_2, y_2) <==> (upper_left, lower_right)
    box2: (x_1, y_1, x_2, y_2) <==> (upper_left, lower_right)
    """
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the intersect area
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    intersect_width = xi2 - xi1
    intersect_height = yi2 - yi1
    intersect_area = max(intersect_width, 0) * max(intersect_height, 0) # max with 0 because they may not intersect

    # Calculate union area
    # formula: Union(A,B) = A + B - intersect_area(A,B)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - intersect_area

    # compute the IoU
    iou = intersect_area / union_area
    return iou

def Exercise_2() -> None:
    # Test case 1
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4)

    print("iou for intersecting boxes = " + str(iou(box1, box2)))
    assert iou(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
    assert np.isclose(iou(box1, box2), 0.14285714), "Wrong value. Check your implementation. Problem with intersecting boxes"

    ## Test case 2: boxes do not intersect
    box1 = (1,2,3,4)
    box2 = (5,6,7,8)
    print("iou for non-intersecting boxes = " + str(iou(box1,box2)))
    assert iou(box1, box2) == 0, "Intersection must be 0"

    ## Test case 3: boxes intersect at vertices only
    box1 = (1,1,2,2)
    box2 = (2,2,3,3)
    print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))
    assert iou(box1, box2) == 0, "Intersection at vertices must be 0"

    ## Test case 4: boxes intersect at edge only
    box1 = (1, 1, 3, 3)
    box2 = (2, 3, 3, 4)
    print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))
    assert iou(box1, box2) == 0, "Intersection at edges must be 0"

    print("\033[92m All tests passed!")
    return None


#############################################################################################################
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Inp
        scores: (None,)
        boxes: (None, 4)
        classes: (None,)
        max_boxes: hyperpara
        iou_threshold: hyperpara

    Out
        scores: predicted score for each box
            -> (None, )
        boxes: predicted box coordinates
            -> (None, 4)
        classes: predicted class for each box
            -> (None, )

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores, iou_threshold=iou_threshold,
                                               max_output_size=tf.Variable(initial_value=max_boxes, dtype='int32'))

    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    # tf.gather == np.where
    scores = tf.gather(params=scores, indices=nms_indices)
    boxes = tf.gather(params=boxes, indices=nms_indices)
    classes = tf.gather(params=classes, indices=nms_indices)
    return scores, boxes, classes


def Exercise_3() -> None:
    tf.random.set_seed(10)
    scores = tf.random.normal(shape=[54, ], mean=1, stddev=4, seed=1)
    boxes = tf.random.normal(shape=[54, 4], mean=1, stddev=4, seed=1)
    classes = tf.random.normal(shape=[54, ], mean=1, stddev=4, seed=1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    assert scores.shape == (10,), "Wrong shape"
    assert boxes.shape == (10, 4), "Wrong shape"
    assert classes.shape == (10,), "Wrong shape"

    assert np.isclose(scores[2].numpy(), 8.147684), "Wrong value on scores"
    assert np.allclose(boxes[2].numpy(), [6.0797963, 3.743308, 1.3914018, -0.34089637]), "Wrong value on boxes"
    assert np.isclose(classes[2].numpy(), 1.7079165), "Wrong value on classes"

    print("\033[92m All tests passed!")
    return None


##############################################################################################################
def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners"""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return kr.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to
    predicted boxes along with their scores, box coordinates and classes.

    Inp:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes: maximum # predicted boxes you'd like
    score_threshold: for getting rid of boxes having prob < score_threshold
    iou_threshold: threshold used for NMS

    Out
        scores -- tensor of shape (None, ), predicted score for each box
        boxes -- tensor of shape (None, 4), predicted box coordinates
        classes -- tensor of shape (None,), predicted class for each box
    """
    # Retrieve outputs of YOLO
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Ignore boxes having confidence score < score_threshold
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape
    boxes = scale_boxes(boxes, image_shape)

    # NMS
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes, iou_threshold=iou_threshold)
    return scores, boxes, classes


def Exercise_4() -> None:
    tf.random.set_seed(10)
    box_xy = tf.random.normal(shape=[19, 19, 5, 2], mean=1, stddev=4, seed=1)
    box_wh = tf.random.normal(shape=[19, 19, 5, 2], mean=1, stddev=4, seed=1)
    box_confidence = tf.random.normal(shape=[19, 19, 5, 1], mean=1, stddev=4, seed=1)
    box_class_prob = tf.random.normal(shape=[19, 19, 5, 80], mean=1, stddev=4, seed=1)

    yolo_outputs = (box_xy, box_wh,  box_confidence, box_class_prob)
    scores, boxes, classes = yolo_eval(yolo_outputs)

    assert scores.shape == (10,), "Wrong shape"
    assert boxes.shape == (10, 4), "Wrong shape"
    assert classes.shape == (10,), "Wrong shape"
    print("\033[92m All tests passed!")
    return None


###############################################################################################################
def predict(yolo_model, img_name):
    """
    Runs the graph to predict boxes for "image_file". Prints and plots the predictions.

    Inp:
    image_file

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """
    # read anchors
    anchors = read_anchors(anchors_path="yolo_v2_608_608/anchors.txt")

    # read class names
    class_names = read_classes(classes_path="yolo_v2_608_608/coco_classes.txt")

    # Preprocess your image
    image, image_data = preprocess_image("Images/" + img_name, model_image_size=(608, 608))

    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))

    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), "Images/" + img_name))

    # Generate colors for drawing bounding boxes.
    colors = get_colors_for_classes(len(class_names))

    # Draw bounding boxes on the image file
    #draw_boxes2(image, out_scores, out_boxes, out_classes, class_names, colors, image_shape)
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)

    # Save the predicted bounding box on the image
    image.save(os.path.join("out", img_name), quality=100)

    # Display the results in the notebook
    output_image = Image.open(os.path.join("out", img_name))
    cv.imshow(output_image)

    return out_scores, out_boxes, out_classes


###############################################################################################################
def main() -> None:
    """
                                            Inputs and outputs of YOLO
    Input: batch of imgs has shape (608, 608, 3)

    Output:
        encoding tensor with the shape of (m, encoded_tensor) where
        m: batch size
        encoded_tensor: shape (nH, nW, anchors, classes) where
            anchors includes:
                pc: prob that there is an object
                bx: bb's x centroid
                by: bb's y centroid
                bh: bb's height
                bw: bb's width
                * bb: bounding box
            classes: prob that the object is a certain class
    """
    # Exercise_1()
    # Exercise_2()
    # Exercise_3()
    # Exercise_4()
    #################################################################################################################
    #                                    Test YOLO Pre-trained Model on Images
    yolo_model = kr.models.load_model(filepath="yolo_v2_608_608/model.h5", compile=False)
    # yolo_model.summary()
    predict(yolo_model=yolo_model, img_name="dog.jpg")

    return None


if __name__ == '__main__':
    main()
