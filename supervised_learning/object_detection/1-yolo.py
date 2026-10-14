#!/usr/bin/env python3
"""
Contains the Yolo class
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Yolo class for object detection using YOLO v3
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        Args:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path to list of class names used for Darknet model
            class_t: box score threshold for initial filtering
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                     containing all anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet model outputs to extract bounding boxes,
        confidences, and class probabilities.

        Args:
            outputs: list of numpy.ndarrays containing predictions from
                     Darknet model for a single image
            image_size: numpy.ndarray containing image's original size
                        [image_height, image_width]

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                   boxes: list of numpy.ndarrays of shape
                          (grid_height, grid_width, anchor_boxes, 4)
                   box_confidences: list of numpy.ndarrays of shape
                                    (grid_height, grid_width, anchor_boxes, 1)
                   box_class_probs: list of numpy.ndarrays of shape
                                    (grid_height, grid_width, anchor_boxes,
                                     classes)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size[0], image_size[1]
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            box = np.zeros(output[..., :4].shape)

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.arange(grid_width)
            c_y = np.arange(grid_height)
            grid_x, grid_y = np.meshgrid(c_x, c_y)
            grid_x = np.expand_dims(grid_x, axis=-1)
            grid_y = np.expand_dims(grid_y, axis=-1)

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            b_x = (sigmoid(t_x) + grid_x) / grid_width
            b_y = (sigmoid(t_y) + grid_y) / grid_height
            b_w = (p_w * np.exp(t_w)) / input_width
            b_h = (p_h * np.exp(t_h)) / input_height

            x1 = (b_x - (b_w / 2)) * image_width
            y1 = (b_y - (b_h / 2)) * image_height
            x2 = (b_x + (b_w / 2)) * image_width
            y2 = (b_y + (b_h / 2)) * image_height

            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

            conf = sigmoid(output[..., 4:5])
            box_confidences.append(conf)

            probs = sigmoid(output[..., 5:])
            box_class_probs.append(probs)

        return boxes, box_confidences, box_class_probs
