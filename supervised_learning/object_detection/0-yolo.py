#!/usr/bin/env python3
"""YOLO object detection class"""

from tensorflow import keras as K


class Yolo:
    """Uses YOLO v3 to perform object detection"""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        Class constructor

        Args:
            model_path: path to Darknet Keras model
            classes_path: path to class names file
            class_t: box score threshold
            nms_t: IOU threshold for non-max suppression
            anchors: anchor boxes
        """

        # Load model
        self.model = K.models.load_model(model_path)

        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        # Thresholds
        self.class_t = class_t
        self.nms_t = nms_t

        # Anchor boxes
        self.anchors = anchors
