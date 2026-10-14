#!/usr/bin/env python3
"""Yolo class"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """Class constructor"""
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
        """Processes Darknet outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h = image_size[0]
        image_w = image_size[1]

        input_h = self.model.input_shape[1]
        input_w = self.model.input_shape[2]

        for i, output in enumerate(outputs):
            grid_h = output.shape[0]
            grid_w = output.shape[1]

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            c_x = np.arange(grid_w)
            c_y = np.arange(grid_h)

            c_x, c_y = np.meshgrid(c_x, c_y)

            c_x = np.expand_dims(c_x, axis=-1)
            c_y = np.expand_dims(c_y, axis=-1)

            bx = (self.sigmoid(tx) + c_x) / grid_w
            by = (self.sigmoid(ty) + c_y) / grid_h

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            bw = (np.exp(tw) * anchor_w) / input_w
            bh = (np.exp(th) * anchor_h) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            box_confidence = self.sigmoid(output[..., 4:5])
            box_confidences.append(box_confidence)

            box_class_prob = self.sigmoid(output[..., 5:])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes,
                     box_confidences,
                     box_class_probs):
        """Filters boxes"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_probs in zip(
                boxes, box_confidences, box_class_probs):

            scores = confidence * class_probs

            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes,
                            box_classes, box_scores):
        """
        Applies non-max suppression

        Returns:
        - box_predictions
        - predicted_box_classes
        - predicted_box_scores
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]
            cls_classes = box_classes[idxs]

            ranked = np.argsort(cls_scores)[::-1]

            cls_boxes = cls_boxes[ranked]
            cls_scores = cls_scores[ranked]
            cls_classes = cls_classes[ranked]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls_classes[0])
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)

                intersection = inter_w * inter_h

                area1 = (
                    (cls_boxes[0, 2] - cls_boxes[0, 0]) *
                    (cls_boxes[0, 3] - cls_boxes[0, 1])
                )

                area2 = (
                    (cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                    (cls_boxes[1:, 3] - cls_boxes[1:, 1])
                )

                union = area1 + area2 - intersection

                iou = intersection / union

                keep = np.where(iou < self.nms_t)[0]

                cls_boxes = cls_boxes[keep + 1]
                cls_scores = cls_scores[keep + 1]
                cls_classes = cls_classes[keep + 1]

        return (
            np.array(box_predictions),
            np.array(predicted_box_classes),
            np.array(predicted_box_scores)
        )
