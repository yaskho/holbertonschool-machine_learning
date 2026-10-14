#!/usr/bin/env python3
"""Yolo class"""

import tensorflow.keras as K
import numpy as np
import cv2
import glob


class Yolo:
    """YOLO v3 object detection class"""

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

    def sigmoid(self, x):
        """
        Sigmoid activation
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet outputs
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

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Grid coordinates
            c_x = np.tile(
                np.arange(grid_w),
                grid_h
            ).reshape(grid_h, grid_w)

            c_y = np.tile(
                np.arange(grid_h).reshape(-1, 1),
                grid_w
            )

            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=2)
            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=2)

            # Box center coordinates
            b_x = (self.sigmoid(t_x) + c_x) / grid_w
            b_y = (self.sigmoid(t_y) + c_y) / grid_h

            # Box width and height
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            b_w = (np.exp(t_w) * anchor_w) / input_w
            b_h = (np.exp(t_h) * anchor_h) / input_h

            # Corner coordinates
            x1 = (b_x - (b_w / 2)) * image_w
            y1 = (b_y - (b_h / 2)) * image_h
            x2 = (b_x + (b_w / 2)) * image_w
            y2 = (b_y + (b_h / 2)) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            # Confidence
            box_confidence = self.sigmoid(output[..., 4:5])
            box_confidences.append(box_confidence)

            # Class probabilities
            box_class_prob = self.sigmoid(output[..., 5:])
            box_class_probs.append(box_class_prob)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes using box scores
        """

        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):

            scores = box_confidences[i] * box_class_probs[i]

            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes,
                            box_classes, box_scores):
        """
        Perform non-max suppression
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

            order = np.argsort(cls_scores)[::-1]

            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]
            cls_classes = cls_classes[order]

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

    @staticmethod
    def load_images(folder_path):
        """
        Load images from a folder
        """

        image_paths = glob.glob(folder_path + '/*')

        images = []

        for path in image_paths:
            image = cv2.imread(path)
            images.append(image)

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Preprocess images for Darknet model
        """

        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:

            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            resized = resized / 255.0

            pimages.append(resized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)
