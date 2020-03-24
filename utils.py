"""
Mask R-CNN
Common utility functions and classes.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion



############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios

    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space

    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride



    # Enumerate combinations of shifts, widths
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)


    # Reshape to get a list of [x] and a list of [w]
    box_centers =  box_centers_x.reshape(-1, 1)
    box_sizes =  box_widths.reshape(-1, 1)

    # Convert to corner coordinates (x1, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, ( x1,  x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)

def norm_boxes(boxes, w):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, ( x1, x2)] in pixel coordinates
    w: width of window
    Note: In pixel coordinates (x2) is outside the box. But in normalized
    coordinates it's inside the box.
    Returns:
        [N, ( x1, x2)] in normalized coordinates
    """

    scale = np.array([w - 1, w - 1])
    shift = np.array([ 0, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)

# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.
    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result



def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (x1, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)


    width = box[:, 1] - box[:, 0]

    center_x = box[:, 0] + 0.5 * width

    gt_width = gt_box[:, 1] - gt_box[:, 0]

    gt_center_x = gt_box[:, 0] + 0.5 * gt_width

    dx = (gt_center_x - center_x) / width

    dw = tf.log(gt_width / width)

    result = tf.stack([dx, dw], axis=1)
    return result


def mask_refinement_graph(box, gt_mask):
    """Compute refinement needed to transform box to gt_box.
    box  [N, (x1, x2)]

    gt_mask [N,num_stations,(x1,x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_mask = tf.cast(gt_mask, tf.float32)


    width = tf.expand_dims((box[:, 1] - box[:, 0]),1)

    center_x = tf.expand_dims(box[:, 0],1) + 0.5 * width

    gt_width = gt_mask[:,:, 1] - gt_mask[:,:, 0]

    gt_center_x = gt_mask[:,:, 0] + 0.5 * gt_width

    dx = (gt_center_x - center_x) / width

    dw = tf.log(gt_width / width)

    result = tf.stack([dx, dw], axis=2)
    return result

############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 2], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]

        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]

            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2 = 0, 0
        boxes[i] = np.array([x1, x2])
    return boxes.astype(np.int32)

def extract_association_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[0], 2], dtype=np.int32)
    for i in range(mask.shape[0]):
        m = mask[i]
        # Bounding box.

        x1= np.min(mask[i,:,0][np.where(mask[i,:,0])[0]]) if np.sum(mask[i,:,0])>0 else 0

        x2= np.max(mask[i,:,1][np.where(mask[i,:,1])[0]]) if np.sum(mask[i,:,1])>0 else 0
        boxes[i] = np.array([x1, x2])
    return boxes.astype(np.int32)

def extract_multi_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 2], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]

        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]

            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2 = 0, 0
        boxes[i] = np.array([x1, x2])
    return boxes.astype(np.int32)



def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, ( x1, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = boxes1[:, 1] - boxes1[:, 0]
    area2 = boxes2[:, 1] - boxes2[:, 0]

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas

    x1 = np.maximum(box[0], boxes[:, 0])
    x2 = np.minimum(box[1], boxes[:, 1])
    intersection = np.maximum(x2 - x1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, ( x1,  x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    Returns:
        [N, ( x1, x2)] in pixel coordinates
    """
    w = shape
    scale = np.array([ w - 1, w - 1])
    shift = np.array([0,  1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def unmold_mask(mask, bbox, window_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(window_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().
    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)
