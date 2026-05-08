#!/usr/bin/env python3
"""Yolo class"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        Class constructor

        Parameters:
        - model_path: path to Darknet Keras model
        - classes_path: path to classes file
        - class_t: box score threshold
        - nms_t: IOU threshold for non-max suppression
        - anchors: anchor boxes
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Calculates sigmoid"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet outputs

        Parameters:
        - outputs: list of numpy.ndarrays
        - image_size: original image size [height, width]

        Returns:
        - boxes
        - box_confidences
        - box_class_probs
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

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            cx = np.arange(grid_w)
            cy = np.arange(grid_h)

            cx, cy = np.meshgrid(cx, cy)

            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = (pw * np.exp(tw)) / input_w
            bh = (ph * np.exp(th)) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            confidence = self.sigmoid(output[..., 4:5])
            box_confidences.append(confidence)

            class_probs = self.sigmoid(output[..., 5:])
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences,
                     box_class_probs):
        """
        Filters boxes using objectness score and class probabilities

        Parameters:
        - boxes: list of processed boxes
        - box_confidences: list of box confidences
        - box_class_probs: list of class probabilities

        Returns:
        - filtered_boxes
        - box_classes
        - box_scores
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_prob in zip(
                boxes, box_confidences, box_class_probs):

            scores = confidence * class_prob

            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
