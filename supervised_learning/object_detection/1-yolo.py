#!/usr/bin/env python3
"""YOLO object detection"""

import numpy as np
from tensorflow import keras as K


class Yolo:
    """YOLO v3 object detector"""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        Class constructor
        """

        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet outputs

        Args:
            outputs: list of predictions
            image_size: original image size

        Returns:
            boxes, box_confidences, box_class_probs
        """

        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h = image_size[0]
        image_w = image_size[1]

        for i, output in enumerate(outputs):

            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchor_boxes = output.shape[2]

            # Extract predictions
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Grid coordinates
            c_x = np.tile(np.arange(grid_w), grid_h)
            c_x = c_x.reshape(grid_h, grid_w)

            c_y = np.tile(np.arange(grid_h).reshape(-1, 1), grid_w)

            c_x = np.repeat(c_x[..., np.newaxis],
                            anchor_boxes, axis=2)
            c_y = np.repeat(c_y[..., np.newaxis],
                            anchor_boxes, axis=2)

            # YOLO formulas
            b_x = (self.sigmoid(t_x) + c_x) / grid_w
            b_y = (self.sigmoid(t_y) + c_y) / grid_h

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            b_w = (anchor_w * np.exp(t_w)) / input_w
            b_h = (anchor_h * np.exp(t_h)) / input_h

            # Convert to corner coordinates
            x1 = (b_x - b_w / 2) * image_w
            y1 = (b_y - b_h / 2) * image_h
            x2 = (b_x + b_w / 2) * image_w
            y2 = (b_y + b_h / 2) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)

            # Confidence
            box_confidence = self.sigmoid(output[..., 4:5])
            box_confidences.append(box_confidence)

            # Class probabilities
            box_class_prob = self.sigmoid(output[..., 5:])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
