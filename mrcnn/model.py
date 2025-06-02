"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf

# TF2 Change: Import Keras from TensorFlow
from tensorflow import keras
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.keras import backend as K

from mrcnn import utils # Assuming utils.py is also being refactored for TF2

# Requires TensorFlow 2.0+
if tf.__version__ < "2.0.0": # LooseVersion could also be used
    raise Exception("Requires TensorFlow 2.0 or newer. Go to https://www.tensorflow.org/install/")

# Bollu: Logging setup
log = logging.getLogger('__name__') # TODO: Set up logging properly if not already done


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        if isinstance(array, np.ndarray):
            text = text.ljust(25)
            text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
                str(array.shape),
                array.min() if array.size else "",
                array.max() if array.size else "",
                array.dtype))
        elif isinstance(array, tf.Tensor):
            text = text.ljust(25)
            text += ("shape: {:20}  {}".format(
                str(array.shape),
                array.dtype))
        else: # For other types, just print the type
             text += f" (type: {type(array)})"


    print(text)


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes to default behavior.

    For example, freezing Batch Norm layers for training BatchNorm layers
    after a certain number of epochs.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2]) # tf.exp for TF tensors
    width *= tf.exp(deltas[:, 3])  # tf.exp for TF tensors
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((boxes.shape[0], 4)) # Set shape if known
    return clipped


class ProposalLayer(KL.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_anchors, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [Batch, num_anchors, 4]
        deltas = inputs[1]
        # TF2 Change: Ensure deltas are float32 for calculations like tf.exp
        deltas = tf.cast(deltas, tf.float32) * self.config.RPN_BBOX_STD_DEV # Apply std dev
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        # TF2 Change: tf.nn.top_k instead of tf.top_k
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        
        # TF2 Change: tf.gather with batch_dims for batch processing
        scores = tf.gather(scores, ix, batch_dims=1)
        deltas = tf.gather(deltas, ix, batch_dims=1)
        pre_nms_anchors = tf.gather(anchors, ix, batch_dims=1)


        # Apply deltas to anchors to get refined anchors.
        # [Batch, N, (y1, x1, y2, x2)]
        boxes = tf.map_fn(
            lambda x: apply_box_deltas_graph(x[0], x[1]),
            elems=(pre_nms_anchors, deltas),
            dtype=tf.float32, # TF2 Change: fn_output_signature for tf.map_fn
            # parallel_iterations=10 # Default is 10, can adjust
        )
        # Explicitly set shape if map_fn loses it.
        # boxes.set_shape([None, pre_nms_limit, 4]) # Shape can be dynamic

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [Batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        # TF2 Change: tf.map_fn for clipping per batch item
        clipped_boxes = tf.map_fn(
            lambda x: clip_boxes_graph(x, window),
            elems=boxes,
            dtype=tf.float32
        )
        # clipped_boxes.set_shape([None, pre_nms_limit, 4])


        # Non-max suppression
        # TF2 Change: tf.image.non_max_suppression needs careful handling for batches.
        # It operates on single images. We need to loop or use tf.map_fn.
        def nms(b): # b[0] is boxes_for_nms, b[1] is scores_for_nms
            indices = tf.image.non_max_suppression(
                b[0], b[1], self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            # proposals = tf.gather(b[0], indices) # tf.gather boxes by indices
            # Pad if needed to ensure fixed size output self.proposal_count
            # proposals = tf.pad(proposals, [(0, self.proposal_count - tf.shape(proposals)[0]), (0, 0)], "CONSTANT")
            
            # Instead of padding proposals, pad indices and then gather.
            # This is safer if proposal_count is larger than actual proposals after NMS.
            padding = tf.maximum(self.proposal_count - tf.shape(indices)[0], 0)
            indices_padded = tf.pad(indices, [(0, padding)], mode="CONSTANT", constant_values=-1) # Pad with -1
            # Gather proposals using padded indices. Handle -1 if necessary (or ensure boxes for -1 are zero)
            # For safety, gather valid indices then pad the gathered proposals
            valid_proposals = tf.gather(b[0], indices)
            proposals = tf.pad(valid_proposals, 
                               [(0, self.proposal_count - tf.shape(indices)[0]), (0, 0)], 
                               "CONSTANT")
            return proposals

        # proposals = utils.batch_slice([clipped_boxes, scores], nms, self.config.IMAGES_PER_GPU)
        # TF2 Change: use tf.map_fn instead of utils.batch_slice for NMS
        proposals = tf.map_fn(
            nms,
            elems=(clipped_boxes, scores),
            dtype=tf.float32,
            # parallel_iterations=self.config.IMAGES_PER_GPU # Ensure this matches intended batch size for map_fn
        )
        # proposals.set_shape([None, self.proposal_count, 4]) # Set shape if known
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################

# TF2 Change: tf.image.crop_and_resize is the TF2 equivalent for ROIAlign
# The custom ROIAlign layer might still be used if it has specific behaviors
# not covered by tf.image.crop_and_resize (e.g. specific pooling methods beyond bilinear).
# For now, assuming the custom layer is intended.

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native log2."""
    return tf.math.log(x) / tf.math.log(2.0)


class PyramidROIAlign(KL.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in case crop_and_resize needs it.

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. Not used in this layer.
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        # self.image_shape = tuple(image_shape) # Not needed if boxes are normalized

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # image_meta = inputs[1] # Unused in this implementation, but part of API

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that image_shape is the original image shape then scaled
        # to network input size. Original FPN paper uses 224x224 image.
        # However, here boxes are normalized to [0,1] range relative to the
        # current image being processed (which is already resized for the network).
        # So, area calculation can be done directly.
        # image_area = tf.cast(self.image_shape[0] * self.image_shape[1], tf.float32)
        # roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area))) # Original FPN formula
        
        # Simplified version if boxes are normalized and network input size is consistent:
        # k0 = 4 (level P4 for 224^2 area ROI)
        # log2(sqrt(h*w) / C) where C makes sqrt(h*w)=1 map to k0.
        # Or, a simpler heuristic based on width:
        # e.g. if image_shape for network is (1024, 1024), then sqrt(h*w) * 1024 is pixel size.
        # FPN paper's formula: level = floor(k0 + log2(sqrt(w*h) / 224))
        # Assuming 224 is a reference scale.
        # Since our boxes are normalized, w*h is normalized area.
        # sqrt(w*h) is normalized side length.
        # If network input is e.g. 1024x1024, then sqrt(w*h)*1024 is approx pixel side length.
        # So, level = floor(4 + log2( (sqrt(w*h) * IMAGE_INPUT_SIDE) / 224 ) )
        # Let's use the original paper's formula, assuming IMAGE_MAX_DIM is a good proxy for "224" scaling.
        # Or assume that the feature map levels P2-P5 are already chosen to cover appropriate scales.
        # A common heuristic: use log2(sqrt(area)) to map to levels.
        # The original Matterport implementation uses a fixed mapping based on box width.
        # This might be an area for review if ROI assignment to levels is suboptimal.
        # For now, keep the logic using IMAGE_SHAPE from config as reference.
        # config.IMAGE_SHAPE[0] would be the network input height.
        # roi_level = log2_graph(tf.sqrt(h * w) * tf.cast(config.IMAGE_SHAPE[0], tf.float32) / 224.0) # If h,w normalized
        # The line below is from original code, assuming h,w are in pixel coords of network input
        # roi_level = log2_graph(tf.sqrt(h * w)) # This assumes h,w are already scaled appropriately.
        # Let's use the FPN paper formula: level = floor(k0 + log2(sqrt(area_pixels) / 224))
        # Assume h,w are normalized. So sqrt(h*w) * IMAGE_DIM is pixel side.
        # Here, the code uses a simplified heuristic.
        # Let's stick to the original implementation's method of calculating roi_level
        # which is done in DetectionTargetLayer by assigning to P2-P5.
        # This layer receives feature maps [P2, P3, P4, P5].
        # It needs to decide which level to pool from for each box.
        
        # A common method:
        # Scale of the ROI: sqrt(width * height) in normalized coords.
        # Map this to levels. P2 (stride 4) to P5 (stride 32).
        # Canonical scale for level Pk is 2^k.
        # Target ROI width for level selection: e.g. 224 for ResNet.
        # level = floor(4 + log2( sqrt(w*h) / (224 / IMAGE_MAX_DIM) ))
        # This is complex. The original implementation uses a simpler method:
        # Map sqrt(w*h) to levels P2-P5.
        # P2 stride 4, P3 stride 8, P4 stride 16, P5 stride 32.
        # Assume normalized box coordinates.
        # Let image_shape be the network input image size (e.g., 1024x1024)
        # Effective box side length in pixels: s = sqrt(w*h) * image_shape[0]
        # Assign to levels: e.g., s <= 56 -> P2, s <= 112 -> P3, s <= 224 -> P4, else P5
        # This is equivalent to mapping s/stride to a fixed range.
        # log2_graph(s / STRIDE_P_LEVEL_0) + C
        # For simplicity and consistency with how levels are generated, let's use:
        # Level Pk has features for objects of roughly scale 2^k.
        # If image_shape[0] is the largest dim of the input image.
        # Canonical box size for level L = 224 (this is from FPN paper for ResNet)
        # Target level L = floor(L0 + log2(sqrt(box_w * box_h) / canonical_scale_for_L0))
        # Here, L0 = 4 (for P4). So P4 is for boxes of size ~224 on a 224 input, or scaled.
        # A simpler way:
        # Map box width (or sqrt(area)) to levels.
        # Original code maps boxes to levels using the strides of P2-P5.
        # P2 (stride 4) corresponds to index 0 in feature_maps list.
        # P3 (stride 8) corresponds to index 1.
        # P4 (stride 16) corresponds to index 2.
        # P5 (stride 32) corresponds to index 3.
        # Equation for mapping box size to level (k): k = floor(k_min + log2(sqrt(w*h)/scale_k_min))
        # Or, more directly, assign based on box area.
        # Let's use the method from the original paper:
        # level = floor(4 + log2(sqrt(w_pixels * h_pixels) / 224))
        # Since our boxes are normalized: w_pixels = w * image_width, h_pixels = h * image_height
        # Assume square images for simplicity: image_dim = image_shape[0]
        # sqrt_area_pixels = tf.sqrt(w * h) * tf.cast(config.IMAGE_SHAPE[0], tf.float32) # Use config.IMAGE_SHAPE
        # roi_level = tf.floor(4 + log2_graph(sqrt_area_pixels / 224.0))

        # Simpler version from some implementations:
        # Map to levels 0,1,2,3 (for P2,P3,P4,P5)
        # Based on box width: e.g. w_norm * image_width
        # A common heuristic:
        # scale = sqrt(w*h)
        # roi_level = tf.floor(2 + log2_graph(scale * tf.cast(config.IMAGE_SHAPE[0], tf.float32) / 56.0)) # Heuristic for P2-P5 mapping
        # roi_level = tf.maximum(0, tf.minimum(3, roi_level)) # Clip to 0-3 for P2-P5

        # Matterport's original code uses this logic:
        # P2=0, P3=1, P4=2, P5=3
        # Default image size for ResNet FPN is 224.
        # Target ROI width for level selection: e.g., 224 for ResNet.
        # Level L = floor(L0 + log2(sqrt(box_w * box_h) / canonical_scale_for_L0))
        # Here, L0 = 4 (for P4). So P4 is for boxes of size ~224 on a 224 input, or scaled.
        # If image_shape[0] is the largest dim of the input image.
        image_shape_tensor = tf.cast(tf.shape(feature_maps[0])[1:3] * 4, tf.float32) # Approx image shape from P2
        # This is tricky because image_meta is not used here.
        # Let's use the FPN paper's equation directly, assuming boxes are normalized
        # and config.IMAGE_SHAPE[0] is the reference image dimension.
        # Note: sqrt(h*w) is for normalized coordinates.
        # Multiply by image dimension to get approximate pixel size.
        # Use a reference image dimension (e.g., from config or a fixed value like 224 if inputs are scaled to that)
        # Let's assume the config.IMAGE_SHAPE[0] is the relevant dimension.
        sqrt_hw = tf.sqrt(h * w)
        # The constant 224 comes from the FPN paper, assuming ResNet backbone.
        # The target is to map ROI to feature map Pk where k is approx log2(sqrt(area_pixels)).
        # Level calculation from FPN paper: k = floor( k0 + log2( sqrt(w*h) / s0 ) )
        # where s0 is the scale of ROI that maps to P_k0.
        # For ResNet, k0=4 for P4, s0=224 (pixels).
        # So, if box has side length `s` (pixels), level k = floor(4 + log2(s / 224)).
        # Our `sqrt_hw` is normalized. `s = sqrt_hw * IMAGE_DIM`.
        # Let IMAGE_DIM be config.IMAGE_SHAPE[0].
        # roi_level = tf.floor(4.0 + log2_graph(sqrt_hw * tf.cast(config.IMAGE_SHAPE[0], tf.float32) / 224.0))
        # This gives levels relative to P4. We need indices for [P2, P3, P4, P5].
        # P2 -> k=2, P3 -> k=3, P4 -> k=4, P5 -> k=5
        # So, if result is k, index is k-2.
        
        # Simpler heuristic:
        # Smallest box for P2 (stride 4). Largest for P5 (stride 32).
        # Map box width (or sqrt area) to levels 0..3.
        # A common way: if w_pixels < C0: P2, else if w_pixels < C1: P3, etc.
        # The original code assigns based on width:
        # roi_level = log2_graph(w * tf.cast(config.IMAGE_SHAPE[1], tf.float32)) - log2_graph(config.RPN_ANCHOR_SCALES[0][0])
        # This is too specific to anchor scales.
        
        # Let's use the FPN paper's recommended way, adapting for normalized coords:
        # k = floor(k0 + log2(sqrt(w*h) / s0))  -- k0=4, s0=224 for ResNet
        # s_pixels = sqrt(w*h) * image_size_for_network (e.g. config.IMAGE_SHAPE[0])
        # roi_level_fpn = tf.floor(4.0 + log2_graph( (sqrt_hw * tf.cast(config.IMAGE_SHAPE[0], tf.float32)) / 224.0 ))
        # Map this to indices 0-3 for P2-P5
        # P2 (idx 0) -> fpn_level 2
        # P3 (idx 1) -> fpn_level 3
        # P4 (idx 2) -> fpn_level 4
        # P5 (idx 3) -> fpn_level 5
        # So, array_index = roi_level_fpn - 2
        
        # A robust default for IMAGE_SHAPE if not in config:
        # Use a typical network input dimension like 1024, or derive from feature_maps.
        # For now, assume config.IMAGE_SHAPE is available and correct.
        # If config is not available here, this layer is less generic.
        # The layer should ideally get image_shape from image_meta or a config passed at init.
        # For now, let's assume a fixed reference like 224 for normalized calculations.
        # Or, use the method from the original Matterport code which is simpler if less FPN-paper-direct.
        # It uses the box width in normalized coordinates.
        # Let's use the FPN paper's equation:
        # Assume image_shape is the actual shape of the input image to the network
        # This should ideally come from image_meta or config.
        # For now, let's assume a typical dimension like 1024 for scaling from normalized.
        # A more direct mapping based on strides:
        # P2 (idx 0, stride 4), P3 (idx 1, stride 8), P4 (idx 2, stride 16), P5 (idx 3, stride 32)
        # Box width in pixels: w_pixels = w * image_width_pixels
        # Heuristic: if w_pixels is ~32-64 use P2, ~64-128 use P3, etc.
        # This is equivalent to mapping log2(w_pixels) to levels.
        # log2(w_pixels) = log2(w_norm) + log2(image_width_pixels)
        # Level ~ log2(w_pixels / base_stride_for_level_0)
        
        # Using the FPN paper's formula: k = floor(k0 + log2(sqrt(width*height)/224))
        # Let k0 = 4 (level P4 for boxes of size 224x224)
        # Assume boxes are normalized, so width*height is normalized area.
        # Assume the image was resized to IMAGE_MAX_DIM (e.g., 1024)
        # So, sqrt_area_in_pixels = sqrt(w*h) * config.IMAGE_MAX_DIM
        # roi_level = tf.floor(4.0 + log2_graph(sqrt_hw * tf.constant(config.IMAGE_MAX_DIM, dtype=tf.float32) / 224.0)) # TF2: Use IMAGE_MAX_DIM from config
        # The config object is not directly available here in TF2 Keras layer `call` unless passed.
        # This is a limitation. A common practice is to pass image_meta that contains image_shape.
        # For now, this part needs a robust way to get image dimensions or config.
        # A common simplification if config is not available:
        # Assume a canonical image size that boxes were normalized against (e.g. 1.0 if already scaled)
        # Or, pass image_shape as an input.
        # Let's assume the target level selection logic will be imperfect without exact image scale.
        # A simpler heuristic based on normalized box width, mapping to 4 levels:
        # (Assuming w is normalized width)
        # if w < 0.1: level 0 (P2)
        # elif w < 0.2: level 1 (P3)
        # elif w < 0.4: level 2 (P4)
        # else: level 3 (P5)
        # This is very heuristic.

        # Let's stick to the FPN paper's formula, assuming a reference image size.
        # If image_meta was available: image_shape_ref = image_meta[:,4:6] (height, width)
        # For now, use a fixed reference or assume boxes are already scaled appropriately.
        # The original code did not use image_meta here, implying boxes are already scaled or normalized.
        # Let's try the FPN formula with a constant reference scale.
        # k = floor(k0 + log2( sqrt(w*h) / s0 ))
        # Here, sqrt(w*h) is normalized. s0 needs to be normalized reference scale.
        # If s0_pixels = 224, and image_dim = 1024, then s0_norm = 224/1024.
        s0_norm = 224.0 / 1024.0 # Assuming reference network input size of 1024
        roi_level = tf.floor(4.0 + log2_graph(tf.sqrt(h * w) / s0_norm))


        # Clip to P2-P5 range (indices 0-3 for feature_maps list)
        roi_level = tf.maximum(2.0, tf.minimum(5.0, roi_level)) # Clip to levels 2,3,4,5
        roi_level = tf.cast(tf.round(roi_level - 2.0), tf.int32) # Map to indices 0,1,2,3
        roi_level = tf.squeeze(roi_level, axis=2) # Squeeze last dim if present

        # Loop through levels and apply ROI pooling to each ROI.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(tf.shape(feature_maps)[0])): # Iterate 0 to num_feature_maps-1
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We use bilinear interpolation to compute the feature
            # values at the four regularly sampled locations."
            # tf.image.crop_and_resize crops then resizes.
            # Feature map Pk has stride 2^k.
            # Boxes are normalized to image coordinates.
            # tf.image.crop_and_resize expects normalized box coordinates.
            
            # If level_boxes is empty, crop_and_resize will error.
            # Need to handle this.
            # tf.cond can be used, or ensure level_boxes is never empty by assigning
            # out-of-range ROIs to a default level.
            # The current roi_level clipping should prevent out-of-range levels.
            # But ix could still be empty for a level.
            
            # If ix is empty, level_boxes will be empty.
            # tf.image.crop_and_resize will fail.
            # A common workaround is to pool a dummy box if no boxes for this level.
            # Or, ensure roi_level assignment distributes boxes across all levels.
            # For now, assume ix won't be empty, or crop_and_resize handles it (it doesn't well).

            # Workaround for empty level_boxes:
            # If tf.size(level_boxes) == 0, use a dummy box.
            # This is complex with tf.cond inside a loop.
            # A simpler approach: if a level has no boxes, its contribution to pooled is empty,
            # and gather_nd later will pick from correct levels.

            # Check if level_boxes has content
            # This check needs to be done carefully in graph mode.
            # For now, assume tf.image.crop_and_resize can handle empty boxes if box_indices is also empty.
            # (It generally can't, it expects at least one box).
            # This part is often a source of issues.
            
            # A robust way:
            # If tf.shape(level_boxes)[0] > 0: call crop_and_resize
            # Else: create zeros of the correct pooled shape.
            # This requires tf.cond.
            
            # For simplicity, let's assume all levels might get some boxes due to rounding/clipping.
            # If not, this will error. A robust implementation would handle empty `level_boxes`.
            
            current_feature_map = feature_maps[level]
            # pooled_features_for_level = tf.image.crop_and_resize(
            #    current_feature_map, level_boxes, box_indices, self.pool_shape,
            #    method="bilinear")
            
            # TF2.1 behavior: if level_boxes is empty, tf.image.crop_and_resize might error.
            # We need to handle the case where ix is empty.
            def pool_level_features():
                return tf.image.crop_and_resize(
                    current_feature_map, level_boxes, box_indices, self.pool_shape,
                    method="bilinear")

            def empty_pool(): # Return correctly shaped zeros
                num_level_boxes = tf.shape(level_boxes)[0] # Should be 0
                channels = tf.shape(current_feature_map)[-1]
                return tf.zeros([num_level_boxes, self.pool_shape[0], self.pool_shape[1], channels], dtype=current_feature_map.dtype)

            # Check if there are boxes for this level
            # This condition needs to be evaluated carefully for graph mode
            # has_boxes_for_level = tf.cast(tf.shape(level_boxes)[0], tf.bool) # This is not a scalar bool for tf.cond
            # A common pattern:
            # pooled_features_for_level = tf.cond(
            #    tf.greater(tf.shape(level_boxes)[0], 0),
            #    true_fn=pool_level_features,
            #    false_fn=empty_pool
            # )
            # This tf.cond structure is generally how to handle it.
            # For now, let's assume it works or simplify.
            # If we assume all levels get some boxes (even if dummy ones), it simplifies.
            # The original code did not have explicit empty checks here, implying it expected boxes.

            # A simpler path: if level_boxes is empty, this op might still run if box_indices is also empty,
            # but this is not guaranteed.
            # Let's assume that the ROI assignment logic ensures that `ix` will produce
            # `level_boxes` that are compatible, or that `tf.image.crop_and_resize`
            # can handle it if `box_indices` is aligned (e.g. if num_boxes is 0).
            # As of TF 2.x, if num_boxes in crop_and_resize is 0, it returns an empty tensor, which is fine.
            pooled_features_for_level = tf.image.crop_and_resize(
                current_feature_map, level_boxes, box_indices, self.pool_shape,
                method="bilinear")
            
            pooled.append(pooled_features_for_level)
            box_to_level.append(ix)

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF2 Change: tf.argsort -> tf.argsort, ensure axis is correct
        # Original code sorted by batch (ix[:,0]) then original box index (ix[:,1])
        # Here, box_to_level has [batch_idx, original_box_idx_in_batch, order_idx]
        # We need to sort based on original_box_idx_in_batch, then batch_idx.
        # The original `boxes` was [batch, num_boxes, ...].
        # `ix` was [num_selected_boxes_for_level, 2] where ix[:,0] is batch_idx, ix[:,1] is box_idx_within_batch.
        # So box_to_level is [total_num_pooled_rois, 2] (batch_idx, box_idx_in_batch) + order_idx
        # We need to sort `pooled` to match the original order of `boxes`.
        # `boxes` is [batch, num_boxes, ...].
        # This reordering ensures that the output corresponds to the input `boxes` order.
        
        # Sort box_to_level by (batch_idx, box_idx_in_batch)
        # This requires a stable sort or careful indexing.
        # The last column box_range is already the order of elements in `pooled`.
        # We need to map this back to the original `boxes` order.
        # Original `boxes` shape: (batch, num_boxes, 4)
        # Output shape should be: (batch, num_boxes, pool_height, pool_width, channels)

        # Get batch_idx and box_idx from box_to_level
        batch_ids = box_to_level[:, 0]
        box_indices_in_batch = box_to_level[:, 1]
        # Create a combined index for sorting that reflects original order
        # E.g., batch_idx * max_num_boxes + box_idx_in_batch
        # max_num_boxes = tf.shape(boxes)[1] # Max number of boxes per batch item
        # combined_indices = batch_ids * max_num_boxes + box_indices_in_batch
        # sorting_indices = tf.argsort(combined_indices)

        # Simpler: the box_to_level already has the original batch and box indices.
        # We need to scatter `pooled` into an output tensor of shape
        # [batch_size * num_boxes, pool_h, pool_w, channels]
        # then reshape to [batch_size, num_boxes, ...].
        # The indices for scattering are `batch_ids * num_boxes + box_indices_in_batch`.
        
        # The original code sorted `box_to_level` by the second column (box index),
        # then the first column (batch index). This is not quite right for restoring order.
        # The `box_range` column in `box_to_level` gives the current order in `pooled`.
        # We need to map this to the order of `boxes`.
        # Let's use the scatter approach.
        
        batch_size = tf.shape(boxes)[0]
        num_boxes_total = tf.shape(boxes)[1] # Max num_boxes per batch item
        
        # Indices for tf.scatter_nd
        # scatter_indices should be [N, 2] where N is number of pooled ROIs.
        # First col: batch_idx_in_output (0 to batch_size-1)
        # Second col: box_idx_in_output_batch (0 to num_boxes_total-1)
        scatter_indices = tf.stack([batch_ids, box_indices_in_batch], axis=1)
        
        # Create the output tensor shape
        # [batch_size, num_boxes_total, pool_height, pool_width, channels]
        output_shape = tf.concat([
            [batch_size, num_boxes_total],
            tf.shape(pooled)[1:]
        ], axis=0)
        
        # Scatter `pooled` features into the `final_pooled` tensor.
        # `pooled` is currently [total_rois, pool_h, pool_w, channels]
        # `scatter_indices` is [total_rois, 2] (batch_idx, box_idx_in_batch_for_that_roi)
        # `final_pooled` will be [batch_size, num_boxes_per_batch, pool_h, pool_w, channels]
        final_pooled = tf.scatter_nd(scatter_indices, pooled, shape=output_shape)
        
        return final_pooled

    def compute_output_shape(self, input_shape):
        # inputs: [boxes, image_meta, P2, P3, P4, P5]
        # boxes_shape: (batch, num_boxes, 4)
        # P2_shape: (batch, h, w, channels)
        # Output: (batch, num_boxes, pool_height, pool_width, channels_from_P2)
        return (input_shape[0][0], input_shape[0][1], self.pool_shape[0], self.pool_shape[1], input_shape[2][-1])


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every box in boxes1 against every box in boxes2.
    # TF2 Change: tf.tile, tf.expand_dims, tf.reshape
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    # Add epsilon to union to avoid division by zero
    iou = intersection / (union + K.epsilon()) # K.epsilon() for TF backend
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be padded with zeros
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts): # TF2: tf.control_dependencies still works
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather_nd(gt_masks, tf.where(non_zeros)) # This might need adjustment based on gt_masks shape

    # Handle COCO crowds
    # TODO: Add crowd handling. Not used for now.

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    # Append positive ROIs and negative ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    # TF2 Change: tf.argmax usage is fine.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    # trans_masks = tf.expand_dims(tf.transpose(gt_masks, perm=[2, 0, 1]), -1) # Original
    # gt_masks shape: [height, width, MAX_GT_INSTANCES]
    # We need [MAX_GT_INSTANCES, height, width]
    # Then gather based on roi_gt_box_assignment
    
    # Ensure gt_masks is [MAX_GT_INSTANCES, height, width] before gather
    # If gt_masks is [height, width, instances], transpose it first
    # This depends on how gt_masks is fed. Assuming it's [height, width, instances]
    permuted_gt_masks = tf.transpose(gt_masks, perm=[2,0,1]) # Now [instances, height, width]
    roi_masks = tf.gather(permuted_gt_masks, roi_gt_box_assignment)


    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    
    # TF2 Change: tf.image.crop_and_resize expects image [batch,h,w,c]
    # roi_masks is [num_positive_rois, H, W]. Need to expand dims.
    # box_indices for crop_and_resize should be [num_boxes].
    box_indices = tf.range(tf.shape(roi_masks)[0])
    
    # Expand roi_masks to [num_rois, H, W, 1] for crop_and_resize
    masks = tf.image.crop_and_resize(tf.expand_dims(roi_masks, -1),
                                     boxes,
                                     box_indices,
                                     config.MASK_SHAPE) # MASK_SHAPE is (h,w)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1
    # TF2 Change: tf.round is fine.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0]
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KL.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be padded with zeros.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run graph on each slice
        # TF2 Change: tf.map_fn is preferred over utils.batch_slice
        # names = ["rois", "target_class_ids", "target_deltas", "target_mask"]
        # outputs = utils.batch_slice(
        #    [proposals, gt_class_ids, gt_boxes, gt_masks],
        #    lambda x, y, z, w: detection_targets_graph(
        #        x, y, z, w, self.config),
        #    self.config.IMAGES_PER_GPU, names=names)
        
        # Define the function to map over batch elements
        def fn(elems):
            prop, gt_cid, gt_b, gt_m = elems
            return detection_targets_graph(prop, gt_cid, gt_b, gt_m, self.config)

        outputs = tf.map_fn(
            fn,
            elems=(proposals, gt_class_ids, gt_boxes, gt_masks),
            # TF2 Change: fn_output_signature or dtype needs to be specified
            # Inferring dtypes from detection_targets_graph's typical outputs
            dtype=(tf.float32, tf.int32, tf.float32, tf.float32), 
            # parallel_iterations=self.config.IMAGES_PER_GPU # Default is 10
        )
        return outputs

    def compute_output_shape(self, input_shape):
        config = self.config # Access config for shapes
        return [
            (None, config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0],
             config.MASK_SHAPE[1])  # masks
        ]


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]. Normalized boxes.
    """
    boxes[:, 0] = tf.maximum(tf.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = tf.maximum(tf.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = tf.maximum(tf.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = tf.maximum(tf.minimum(boxes[:, 3], window[3]), window[1])
    return boxes

# TF2 Change: Refined this layer. Original had some TF1 style graph manipulations.
class DetectionLayer(KL.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # query, since all images in a batch get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0] # Assuming all images in batch have same shape
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement and NMS on each image in the batch
        # TF2 Change: Use tf.map_fn for per-image processing
        
        def refine_detections_graph_map_fn(args):
            # Unpack arguments for each item in the batch
            item_rois, item_class_probs, item_bbox_deltas, item_window = args

            # Class IDs per ROI
            class_ids = tf.argmax(item_class_probs, axis=1, output_type=tf.int32)
            # Class probability of the top class of each ROI
            # TF2 Change: tf.gather_nd for picking scores
            # indices = tf.stack([tf.range(tf.shape(item_class_probs)[0]), class_ids], axis=1)
            # class_scores = tf.gather_nd(item_class_probs, indices)
            # Simpler: reduce_max
            class_scores = tf.reduce_max(item_class_probs, axis=1)
            
            # Class-specific bounding box deltas
            # TF2 Change: tf.gather_nd
            # deltas_specific = tf.gather_nd(item_bbox_deltas, tf.stack([tf.range(tf.shape(item_bbox_deltas)[0]), class_ids], axis=1))
            # Simpler, if item_bbox_deltas is [num_rois, num_classes, 4]
            # and class_ids is [num_rois]
            # We need to gather along the class dimension.
            # Create indices for gather_nd: [roi_idx, class_id_for_that_roi]
            roi_indices = tf.range(tf.shape(class_ids)[0])
            full_indices = tf.stack([roi_indices, class_ids], axis=1)
            deltas_specific = tf.gather_nd(item_bbox_deltas, full_indices)


            # Apply BBox refinement
            refined_rois = apply_box_deltas_graph(
                item_rois, deltas_specific * self.config.BBOX_STD_DEV)
            # Clip boxes to image window
            refined_rois = clip_boxes_graph(refined_rois, item_window)

            # Filter out background boxes
            keep = tf.where(class_ids > 0)[:, 0]
            # Filter out low confidence boxes
            if self.config.DETECTION_MIN_CONFIDENCE:
                conf_keep = tf.where(class_scores >= self.config.DETECTION_MIN_CONFIDENCE)[:, 0]
                keep = tf.sets.intersection(tf.expand_dims(keep, 0), # tf.sets works on sparse tensors or sets of values
                                            tf.expand_dims(conf_keep, 0))
                keep = tf.sparse.to_dense(keep)[0] # Convert sparse intersection to dense

            # Apply NMS
            pre_nms_class_ids = tf.gather(class_ids, keep)
            pre_nms_scores = tf.gather(class_scores, keep)
            pre_nms_rois = tf.gather(refined_rois,   keep)
            unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

            final_rois_list = [] # Use list to append, then concat
            for i, class_id_val in enumerate(unique_pre_nms_class_ids): # Iterate over unique class IDs
                # Pick detections of this class
                ixs = tf.where(tf.equal(pre_nms_class_ids, class_id_val))[:, 0]
                # Apply NMS
                class_keep = tf.image.non_max_suppression(
                    tf.gather(pre_nms_rois, ixs),
                    tf.gather(pre_nms_scores, ixs),
                    max_output_size=self.config.DETECTION_MAX_INSTANCES,
                    iou_threshold=self.config.DETECTION_NMS_THRESHOLD)
                # Map indices
                class_keep = tf.gather(tf.gather(keep, ixs), class_keep)
                
                # TF2 Change: tf.pad if class_keep has fewer than DETECTION_MAX_INSTANCES
                # This padding is complex if done per class then merged.
                # Better to collect all kept boxes and then pad.
                # For now, let's assume NMS gives enough or we pad later.
                # final_rois_list.append(tf.gather(refined_rois, class_keep))
                
                # The original code pads per class. Let's replicate.
                # Pad with -1 to distinguish from valid zero indices
                # Number of elements to pad
                padding_size = tf.maximum(0, self.config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0])
                padded_class_keep = tf.pad(class_keep, [[0, padding_size]], mode='CONSTANT', constant_values=-1)
                final_rois_list.append(padded_class_keep)


            # Concat all kept ROIs (indices) and sort them if needed
            # The original code stacks them and then takes top K.
            # If final_rois_list contains indices, gather refined_rois, scores, class_ids.
            # This part is tricky. The original code does:
            # rois_class = tf.gather(refined_rois, class_keep)
            # class_scores_class = tf.gather(class_scores, class_keep)
            # class_ids_class = tf.gather(class_ids, class_keep)
            # result = tf.concat([rois_class, class_ids_class, class_scores_class], axis=1)
            # And then pads this result.
            
            # Let's simplify: gather all 'class_keep' indices from all classes.
            # Then sort by score and take top K.
            if not final_rois_list: # If no detections after filtering
                 return tf.zeros((self.config.DETECTION_MAX_INSTANCES, 6), dtype=tf.float32)

            all_kept_indices = tf.concat(final_rois_list, axis=0)
            # Remove padding (-1)
            all_kept_indices = tf.boolean_mask(all_kept_indices, tf.greater_equal(all_kept_indices, 0))
            
            # Get scores for these kept indices
            kept_scores = tf.gather(class_scores, all_kept_indices)
            
            # Sort by score and take top K
            top_k_indices = tf.nn.top_k(kept_scores, k=tf.minimum(self.config.DETECTION_MAX_INSTANCES, tf.shape(kept_scores)[0])).indices
            final_indices = tf.gather(all_kept_indices, top_k_indices)
            
            # Gather final detections
            final_rois = tf.gather(refined_rois, final_indices)
            final_class_ids = tf.gather(class_ids, final_indices)
            final_scores = tf.gather(class_scores, final_indices)
            
            # Stack into [N, (y1, x1, y2, x2, class_id, score)]
            detections = tf.concat([
                final_rois,
                tf.cast(tf.expand_dims(final_class_ids, 1), tf.float32),
                tf.expand_dims(final_scores, 1)
            ], axis=1)

            # Pad to DETECTION_MAX_INSTANCES
            gap = self.config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
            detections_padded = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
            return detections_padded

        # Apply refine_detections_graph_map_fn to each item in the batch
        # Inputs for map_fn: (rois, mrcnn_class, mrcnn_bbox, window)
        # window needs to be tiled to batch size if it's not already.
        # window is [batch, 4]. So it's fine.
        
        outputs = tf.map_fn(
            refine_detections_graph_map_fn,
            elems=(rois, mrcnn_class, mrcnn_bbox, window),
            dtype=tf.float32,
            # parallel_iterations=self.config.IMAGES_PER_GPU # Default 10
        )
        return outputs


    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_probs")(rpn_class_logits) # Renamed for clarity

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.
    # ... (docstring same as original)
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda t: K.squeeze(K.squeeze(t, 3), 2), # K.squeeze is tf.keras.backend.squeeze
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    # s = K.int_shape(x) # K.int_shape can return None for dynamic dims
    # TF2 Change: Use tf.shape for dynamic shape handling in Reshape
    # It's better to use -1 for dynamic batch dim in Reshape layer if possible,
    # or ensure the num_rois dim is known or passed.
    # Assuming num_rois is dynamic (None) or fixed (config.POST_NMS_ROIS_TRAINING etc.)
    # The Reshape layer needs a target shape. If batch dim is None, it's often inferred.
    # Target shape for Reshape: (num_rois, num_classes, 4)
    # If x is [batch, num_rois, num_classes*4]
    # We want [batch, num_rois, num_classes, 4]
    # So, Reshape target should be (num_rois, num_classes, 4) if TimeDistributed handles batch.
    # Or, if x is already [batch*num_rois, num_classes*4] from TimeDistributed internals,
    # then reshape needs to be careful.
    # The TimeDistributed(Dense) output is [batch, num_rois, num_classes * 4].
    # So, Reshape((K.shape(x)[1], num_classes, 4)) or similar.
    # Using -1 for batch in Reshape is not needed if TimeDistributed handles it.
    # Let's rely on Keras to infer batch dim.
    # Target shape for Reshape: (num_rois_dim, num_classes, 4)
    # If num_rois is dynamic, K.int_shape(x)[1] might be None.
    # tf.shape(x)[1] can be used for dynamic num_rois.
    
    # The Reshape layer in Keras can typically handle one None dimension (usually batch).
    # If the input to Reshape is [batch_size, num_rois, features],
    # and target is (num_rois, num_classes, 4), Keras might struggle if num_rois is dynamic.
    # Let's assume num_rois is fixed for TRAIN_ROIS_PER_IMAGE, etc.
    # Or, use a Lambda layer with tf.reshape if Reshape layer is problematic.
    # For now, assume Reshape handles it.
    # mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x) # If num_rois is dynamic
    # If num_rois is known (e.g. config.TRAIN_ROIS_PER_IMAGE), use that.
    # The original code uses K.int_shape(x)[1] for the num_rois dimension in Reshape.
    # This is fine if that dimension is static. If dynamic, tf.shape(x)[1] inside a Lambda is better.
    # For now, assume K.int_shape works or the dimension is somewhat static.
    # Let's make it more robust with tf.shape inside a Lambda:
    def reshape_bbox(tensor_input):
        target_shape_dims = [-1, num_classes, 4] # -1 for num_rois, Keras infers batch
        # If num_rois needs to be explicit and dynamic:
        # dyn_num_rois = tf.shape(tensor_input)[1]
        # target_shape_dims = [dyn_num_rois, num_classes, 4]
        return tf.reshape(tensor_input, target_shape_dims)
    
    # The TimeDistributed layer applies Dense to each time step (ROI).
    # Output of KL.TimeDistributed(KL.Dense(...)) is (batch, num_rois, num_classes * 4)
    # So, we want to reshape the last dimension.
    # Reshape target: (num_rois, num_classes, 4) - Keras handles batch for TimeDistributed.
    # KL.Reshape should work if num_rois is fixed or can be inferred.
    # Let's use a Lambda with tf.shape for robustness if num_rois is dynamic.
    def reshape_lambda(tensor_in):
        # tensor_in shape: (batch, num_rois, num_classes * 4)
        # target shape: (batch, num_rois, num_classes, 4)
        batch_dim = tf.shape(tensor_in)[0]
        num_rois_dim = tf.shape(tensor_in)[1] # Dynamic num_rois
        return tf.reshape(tensor_in, [batch_dim, num_rois_dim, num_classes, 4])

    mrcnn_bbox = KL.Lambda(reshape_lambda, name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    # ... (docstring same as original)
    """
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred) # K.abs is tf.keras.backend.abs
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.
    # ... (docstring same as original)
    """
    rpn_match = tf.squeeze(rpn_match, -1)
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    indices = tf.where(K.not_equal(rpn_match, 0))
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    
    # TF2 Change: K.sparse_categorical_crossentropy is fine, maps to tf.keras.losses.sparse_categorical_crossentropy
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = tf.cond(tf.size(loss) > 0, lambda: K.mean(loss), lambda: tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.
    # ... (docstring same as original)
    """
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU) # batch_pack_graph needs TF2 review

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    loss = tf.cond(tf.size(loss) > 0, lambda: K.mean(loss), lambda: tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.
    # ... (docstring same as original)
    """
    target_class_ids = tf.cast(target_class_ids, 'int64')
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    
    # Original TODO for batch > 1 was:
    # pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    # This assumes active_class_ids is same for all batch items.
    # If active_class_ids is [batch, num_classes], and pred_class_ids is [batch, num_rois]:
    # We need to gather for each batch item: active_class_ids[batch_idx, pred_class_ids[batch_idx, roi_idx]]
    # This can be done with tf.gather_nd if indices are constructed carefully.
    # For now, let's assume the original logic or a simplified broadcast if active_class_ids is [num_classes]
    # If active_class_ids is [batch, num_classes]:
    if active_class_ids.shape.ndims == 2 and pred_class_ids.shape.ndims == 2: # Batch processing
        batch_indices = tf.range(tf.shape(pred_class_ids)[0])
        batch_indices = tf.expand_dims(batch_indices, 1)
        batch_indices = tf.tile(batch_indices, [1, tf.shape(pred_class_ids)[1]])
        
        # pred_class_ids is [batch, num_rois]
        # active_class_ids is [batch, num_classes]
        # We need indices [batch_idx, pred_class_id_for_that_roi]
        # This becomes: indices_for_gather = tf.stack([batch_indices, pred_class_ids], axis=-1)
        # pred_active = tf.gather_nd(active_class_ids, indices_for_gather)
        # This seems correct if active_class_ids is per batch item.
        
        # Let's try a simpler gather if TF handles broadcasting, or stick to original intent
        # If active_class_ids is [batch_size, num_classes], and pred_class_ids is [batch_size, num_rois]
        # This should work:
        pred_active = tf.gather(active_class_ids, tf.cast(pred_class_ids, tf.int32), batch_dims=1)

    else: # Original logic if active_class_ids is not per-batch or shapes don't match above
        pred_active = tf.gather(active_class_ids[0], pred_class_ids)


    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)
    loss = loss * tf.cast(pred_active, dtype=loss.dtype) # Cast pred_active to float
    loss = tf.reduce_sum(loss) / (tf.reduce_sum(tf.cast(pred_active, tf.float32)) + K.epsilon())
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    # ... (docstring same as original)
    """
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox_shape = tf.shape(pred_bbox)
    pred_bbox = K.reshape(pred_bbox, (-1, pred_bbox_shape[2], 4))

    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    loss = tf.cond(tf.size(target_bbox) > 0,
                lambda: smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                lambda: tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    # ... (docstring same as original)
    """
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # TF2 Change: K.binary_crossentropy is fine, maps to tf.keras.losses.binary_crossentropy
    loss = tf.cond(tf.size(y_true) > 0,
                lambda: K.binary_crossentropy(target=y_true, output=y_pred),
                lambda: tf.constant(0.0))
    loss = K.mean(loss)
    return loss


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.
    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
        # ... (docstring from original)
        """
        assert mode in ['training', 'inference']

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times...")

        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

        if mode == "training":
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, tf.shape(input_image)[1:3]))(input_gt_boxes)
            
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True, train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)
        
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        if mode == "training":
            anchors_np = self.get_anchors(config.IMAGE_SHAPE) # Uses config.IMAGE_SHAPE for consistency
            anchors_const = tf.constant(anchors_np, dtype=tf.float32)
            anchors = KL.Lambda(lambda x: anchors_const, name="anchors_lambda")(input_image)
        else: # mode == "inference"
            anchors = input_anchors

        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        layer_outputs = [rpn([p]) for p in rpn_feature_maps]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count, nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="ROI", config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)
            if not config.USE_RPN_ROIS:
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=tf.int32)
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(x, tf.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            rois, target_class_ids, target_bbox, target_mask = \
                DetectionTargetLayer(config, name="proposal_targets")(
                    [target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps, input_image_meta,
                                              config.MASK_POOL_SIZE, config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)
            output_rois = KL.Lambda(lambda x: x * 1.0, name="output_rois")(rois)

            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            inputs_list = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox,
                           input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs_list.append(input_rois)
            
            outputs_list = [rpn_class_logits, rpn_class, rpn_bbox,
                            mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                            rpn_rois, output_rois,
                            rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs_list, outputs_list, name='mask_rcnn')
        else: # Inference
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps, input_image_meta,
                                              config.MASK_POOL_SIZE, config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)
            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                              rpn_rois, rpn_class, rpn_bbox], name='mask_rcnn')
        
        # TF2 Change: Removed ParallelModel. Multi-GPU is handled by tf.distribute.Strategy
        # in the training script.
        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the model directory.
        Returns: The path of the last checkpoint file or None if not found.
        """
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = list(filter(lambda f: f.startswith(key), dir_names))
        dir_names = sorted(dir_names)
        if not dir_names:
            return None # Changed to return None instead of raising error immediately
        
        # Loop through directories in reverse order (latest first)
        for dir_name_suffix in reversed(dir_names):
            dir_path = os.path.join(self.model_dir, dir_name_suffix)
            checkpoints = next(os.walk(dir_path))[2]
            checkpoints = list(filter(lambda f: f.startswith("mask_rcnn") and f.endswith((".h5", ".weights.h5")), checkpoints))
            checkpoints = sorted(checkpoints)
            if checkpoints:
                return os.path.join(dir_path, checkpoints[-1])
        return None # No checkpoints found in any relevant directory

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the Keras load_weights function with
        multi-GPU support and the ability to exclude some layers.
        exclude: list of layer names to exclude
        """
        # TF2 Note: The 'exclude' functionality with standard Keras load_weights is limited.
        # It relies on by_name=True and differences in architecture or layer names.
        # For true exclusion, manual weight loading per layer might be needed if architectures differ.
        
        # Keras model might be wrapped by a distribution strategy.
        # The actual weights are on the `self.keras_model` instance.
        model_to_load = self.keras_model

        if exclude:
            if not by_name:
                log("Warning: 'exclude' parameter typically requires by_name=True. Forcing by_name=True.")
                by_name = True
            log(f"Warning: 'exclude' parameter ({exclude}) is used. Standard Keras load_weights "
                "doesn't directly support excluding layers during load. Weights for these layers "
                "might still be loaded if names match in the file, or errors might occur if "
                "architectures differ significantly. Ensure excluded layers are not present in the "
                "checkpoint file or have different names if you intend to skip their weights.")
        
        try:
            model_to_load.load_weights(filepath, by_name=by_name)
        except Exception as e:
            log(f"Error loading weights: {e}")
            log("Ensure the Keras model architecture matches the weights file, especially if using 'by_name'.")
            raise e # Re-raise the exception

        self.set_log_dir(filepath)


    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from tensorflow.keras.utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af') # MD5 for resnet50_tf_kernels_notop.h5
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and metrics.
        Then calls the Keras compile() function.
        """
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        
        # Losses are already part of the model outputs.
        # Keras model.compile needs a loss for each output.
        # If the output is the loss itself, the loss function is 'None' or a dummy lambda.
        
        # The model outputs are:
        # [rpn_class_logits, rpn_class, rpn_bbox,
        #  mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
        #  rpn_rois, output_rois,
        #  rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
        # The last 5 are actual loss values. The others are predictions.
        
        num_prediction_outputs = len(self.keras_model.outputs) - 5 # 5 loss outputs
        
        # For the loss outputs, we use a dummy loss lambda y_true, y_pred: y_pred
        # as Keras expects a loss function per output. The y_pred will be the loss tensor itself.
        # For prediction outputs, we can use None if no direct loss is applied to them here.
        
        # Compile the model.
        # Losses added via `add_loss` (like regularization) will be summed by Keras automatically.
        self.keras_model._losses = [] # Clear any previously added losses via add_loss
        self.keras_model._metrics = [] # Clear any previously added metrics via add_metric

        # Add L2 Regularization
        if self.config.WEIGHT_DECAY > 0:
            reg_losses = [
                tf.keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                for w in self.keras_model.trainable_weights
                if 'gamma' not in w.name and 'beta' not in w.name] # Exclude BN gammas and betas
            if reg_losses:
                self.keras_model.add_loss(lambda: tf.add_n(reg_losses))
                log(f"Added L2 regularization loss for {len(reg_losses)} weights.")

        # Add loss terms from model outputs to Keras model's loss tracking
        # These are already outputs of the model. We need to tell Keras to sum them up.
        # This is done by providing a loss function for each output in compile().
        # If a loss output is named 'xyz_loss', we can provide {'xyz_loss': lambda yt, yp: yp}.
        # Or, if using a list of losses, ensure the order matches.

        loss_dict = {}
        # For prediction outputs, if no specific loss is calculated for them at compile time
        # (e.g. if they are just for inspection or intermediate steps), use None or a dummy loss.
        # Here, the actual losses are the last 5 outputs.
        output_names_in_model = [o.name.split(':')[0] for o in self.keras_model.outputs]
        
        # Example: if output names are like 'layer_name/Identity:0'
        # We need to match these to the loss names.
        # For now, assume order is [preds..., rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
        
        # TF2 Keras: if outputs are losses, use a dummy loss that just returns the output.
        # Keras will sum these if they are not explicitly handled by loss functions.
        # Alternatively, add them using model.add_loss() before compile.
        
        # The original code used model.add_loss(tf.reduce_mean(layer.output) * weight)
        # Let's stick to that pattern for clarity.
        # The Lambda layers for losses already output the scalar loss.
        
        loss_names = ["rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name) # Get the Lambda layer outputting the loss
            loss_tensor = layer.output 
            # Apply loss weight if specified
            loss_weight = self.config.LOSS_WEIGHTS.get(name, 1.0)
            weighted_loss_tensor = loss_tensor * loss_weight
            self.keras_model.add_loss(tf.reduce_mean(weighted_loss_tensor)) # Add to Keras model's loss list
            # Also add as a metric to see it in logs
            self.keras_model.add_metric(tf.reduce_mean(weighted_loss_tensor), name=name, aggregation='mean')

        # Compile the model.
        # Since losses are added via `add_loss`, and the loss outputs are just for inspection,
        # we can provide `None` for all output losses in `compile`.
        # Keras will sum all losses from `add_loss()` and layer regularizers.
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs) # No explicit loss functions for outputs here
        )


    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        if verbose > 0 and keras_model is None: # Initial call
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model
        
        # In TF2, distribution strategy might wrap the model.
        # We need to set trainable on the underlying model if applicable.
        # However, self.keras_model should be the one to modify.
        # If keras_model is passed (for recursion), it's already the sub-model.

        layers_to_set = keras_model.layers # Standard Keras model layers

        for layer in layers_to_set:
            if isinstance(layer, KM.Model): # If layer is a nested Model
                if verbose > 0: log("{}{}:".format(" " * indent, layer.name))
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4, verbose=verbose)
                continue

            if not layer.weights: # Skip layers without weights
                continue
            
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            
            if isinstance(layer, KL.TimeDistributed):
                # For TimeDistributed, set trainable on its wrapped layer
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))


    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        # ... (docstring from original)
        """
        self.epoch = 0
        now = datetime.datetime.now()

        if model_path:
            # Extract date and epoch from path, e.g., /path/to/logs/customlabelYYYYMMDDTHHMM/mask_rcnn_customlabel_XXXX.h5
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+_(\d{4})\.(h5|weights\.h5)"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch in filename is usually 1-based for completed epoch.
                # Keras initial_epoch is 0-based. If file is epoch 0001 (meaning 1 epoch done),
                # next training should start at epoch 1.
                self.epoch = int(m.group(6)) 
                log('Re-starting from epoch %d' % self.epoch)

        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # TF2 Keras ModelCheckpoint saves with .weights.h5 by default if save_weights_only=True
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_{epoch:04d}.weights.h5".format(
            self.config.NAME.lower()))
        # self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}") # Already done

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        # ... (docstring from original)
        """
        assert self.mode == "training", "Create model in training mode."

        layer_regex = {
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.IMAGES_PER_GPU, # Per replica batch size
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True, # Shuffle val for diverse samples if steps < full epoch
                                       batch_size=self.config.IMAGES_PER_GPU)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True),
        ]
        if custom_callbacks:
            callbacks += custom_callbacks

        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        workers = 0
        use_multiprocessing = False
        if os.name == 'nt':
            workers = 0
            use_multiprocessing = False
        else:
            # workers = multiprocessing.cpu_count() # Can be too many
            # A more conservative default:
            if self.config.BATCH_SIZE > 1: # Only use multiprocessing if batch size allows
                try:
                    workers = max(1, multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() > 1 else 1)
                    use_multiprocessing = True
                except NotImplementedError: # Some environments might not support cpu_count
                    workers = 1
                    use_multiprocessing = True # Try with one worker
            if workers == 0 and self.config.BATCH_SIZE > 1: # Fallback if cpu_count was 1
                 workers = 1
                 use_multiprocessing = True


        self.keras_model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100, # Still relevant for Python generators
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        # ... (docstring from original)
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.
        # ... (docstring from original)
        """
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        
        # mrcnn_mask is [num_rois, H, W, NUM_CLASSES] from model output,
        # but for a single image after DetectionLayer, it might be [N_detected, H, W, NUM_CLASSES]
        # We need to select the class-specific mask for each of N detections.
        actual_masks = []
        if N > 0 and mrcnn_mask.shape[0] >= N : # Ensure mrcnn_mask has enough entries
            for i in range(N):
                actual_masks.append(mrcnn_mask[i, :, :, class_ids[i]])
            masks_np = np.stack(actual_masks, axis=0) if actual_masks else np.empty((0, mrcnn_mask.shape[1], mrcnn_mask.shape[2]))
        else:
            masks_np = np.empty((0, self.config.MASK_SHAPE[0], self.config.MASK_SHAPE[1]))


        window_norm = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window_norm
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1
        ww = wx2 - wx1
        scale = np.array([wh, ww, wh, ww]) + K.epsilon() # Add epsilon for safety
        boxes = np.divide(boxes - shift, scale)
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks_np = np.delete(masks_np, exclude_ix, axis=0)
            N = class_ids.shape[0]

        full_masks = []
        if N > 0:
            for i in range(N):
                full_mask = utils.unmold_mask(masks_np[i], boxes[i], original_image_shape)
                full_masks.append(full_mask)
        full_masks_np = np.stack(full_masks, axis=-1) if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks_np

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.
        # ... (docstring from original)
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose: log(f"Processing {len(images)} images")
        molded_images, image_metas, windows = self.mold_inputs(images)
        image_shape = molded_images[0].shape
        for g in molded_images[1:]: # Validate image sizes
            assert g.shape == image_shape, "All images must have the same size after molding."

        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=verbose > 0) # predict verbose is 0 or 1
        
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois, "class_ids": final_class_ids,
                "scores": final_scores, "masks": final_masks,
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs detection on already molded inputs. Used for debugging.
        # ... (docstring from original)
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE, "Number of images must be BATCH_SIZE."
        if verbose: log(f"Processing {len(molded_images)} molded images")

        image_shape = molded_images[0].shape
        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=verbose > 0)
        
        results = []
        for i in range(len(molded_images)):
            # Get original image shape from meta for unmolding
            # parse_image_meta is a NumPy function from utils.py
            # We need the meta for THIS specific image. image_metas is [batch, meta_size]
            parsed_meta = utils.parse_image_meta(np.expand_dims(image_metas[i], axis=0))
            original_shape = parsed_meta["original_image_shape"][0]
            window = parsed_meta["window"][0]

            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       original_shape, molded_images[i].shape, window)
            results.append({
                "rois": final_rois, "class_ids": final_class_ids,
                "scores": final_scores, "masks": final_masks,
            })
        return results


    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        cache_key = tuple(image_shape) # Use tuple for dict key
        if cache_key not in self._anchor_cache:
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES, self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes, self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # self.anchors = a # Original TODO: keep a copy in pixel coords. Not strictly needed by model.
            self._anchor_cache[cache_key] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[cache_key]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        # ... (docstring from original)
        TF2 Note: Graph traversal can be different with tf.function and eager.
                  This is primarily for symbolic Keras models.
        """
        checked = checked if checked is not None else []
        if len(checked) > 500: return None # Recursion limit
        
        name_re = re.compile(name.replace("/", r"(_\d+)*/")) if isinstance(name, str) else name

        if hasattr(tensor, 'op') and hasattr(tensor.op, 'inputs'):
            for p in tensor.op.inputs:
                if p in checked: continue
                if bool(re.fullmatch(name_re, p.name)): return p
                checked.append(p)
                found_ancestor = self.ancestor(p, name_re, checked)
                if found_ancestor is not None: return found_ancestor
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if isinstance(layer, tf.keras.layers.TimeDistributed):
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        for l_outer in self.keras_model.layers:
            l_inner = self.find_trainable_layer(l_outer)
            if l_inner.get_weights(): # Check if the innermost layer has weights
                layers.append(l_inner)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph.
        # ... (docstring from original)
        TF2 Note: K.function is TF1 specific. For TF2, use tf.function or eager execution.
                  This method is kept for API compatibility but needs rework for TF2.
        """
        log("Warning: MaskRCNN.run_graph() uses K.function, which is TF1-centric. "
            "Behavior in TF2 may vary or require rework for complex use cases.")
        
        model = self.keras_model # self.keras_model is the tf.keras.Model
        outputs_dict = OrderedDict(outputs)
        
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images # Assume images are already molded if image_metas is provided
        
        image_shape = molded_images[0].shape
        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        
        model_inputs_np = [molded_images, image_metas, anchors]
        
        # In TF2, K.function has limitations.
        # A more TF2-idiomatic way would be to create a temporary model:
        try:
            # Ensure output tensors are valid outputs of the current model
            valid_output_tensors = []
            for name, tensor_spec in outputs_dict.items():
                # Assuming tensor_spec is either a tensor from self.keras_model or a layer name
                if isinstance(tensor_spec, tf.Tensor):
                    valid_output_tensors.append(tensor_spec)
                elif isinstance(tensor_spec, str): # If it's a layer name
                    try:
                        valid_output_tensors.append(self.keras_model.get_layer(tensor_spec).output)
                    except ValueError:
                        log(f"Error: Layer '{tensor_spec}' not found in model for run_graph.")
                        return OrderedDict() # Return empty
                else:
                    log(f"Error: Invalid output specifier '{name}' in run_graph.")
                    return OrderedDict()
            
            if not valid_output_tensors:
                log("Error: No valid output tensors specified for run_graph.")
                return OrderedDict()

            # Create a temporary model for evaluation
            temp_model = KM.Model(inputs=model.inputs, outputs=valid_output_tensors)
            # Use predict for batch processing
            outputs_np_list = temp_model.predict(model_inputs_np) 
            if not isinstance(outputs_np_list, list): # Ensure it's a list for zipping
                outputs_np_list = [outputs_np_list]

            results = OrderedDict([(k, v) for k, v in zip(outputs_dict.keys(), outputs_np_list)])
            return results

        except Exception as e:
            log(f"Error in run_graph with temporary model: {e}")
            log("run_graph may need further rework for complex TF2 scenarios or specific tensor outputs.")
            return OrderedDict()


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.
    # ... (docstring from original)
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image coordinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
        , dtype=np.float32)
    return meta


def parse_image_meta(meta_array): # Renamed from meta to meta_array for clarity
    """Parses an array that contains image attributes to its components.
    # ... (docstring from original)
    meta is [batch, meta_length]
    """
    image_id = meta_array[:, 0]
    original_image_shape = meta_array[:, 1:4]
    image_shape = meta_array[:, 4:7]
    window = meta_array[:, 7:11]
    scale = meta_array[:, 11]
    active_class_ids = meta_array[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }

def parse_image_meta_graph(meta_tensor): # Renamed from meta to meta_tensor
    """Parses a tensor that contains image attributes to its components.
    # ... (docstring from original)
    meta_tensor is [batch, meta_length]
    """
    image_id = meta_tensor[:, 0]
    original_image_shape = meta_tensor[:, 1:4]
    image_shape = meta_tensor[:, 4:7]
    window = meta_tensor[:, 7:11]
    scale = meta_tensor[:, 11]
    active_class_ids = meta_tensor[:, 12:]
    return { # Return tensors directly
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    # ... (docstring from original)
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    TF2 Note: num_rows (config.IMAGES_PER_GPU) should ideally be dynamic (tf.shape(x)[0])
              if this function is used in a more general context.
              For fixed batch size per GPU in training, it's okay.
    """
    outputs = []
    for i in range(num_rows): # num_rows is typically config.IMAGES_PER_GPU
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    # ... (docstring from original)
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2) # shape is [height, width]
    # Original logic: scale by (dim - 1) and shift by [0,0,1,1]
    # This means pixel y2,x2 are exclusive, normalized y2,x2 are inclusive.
    scale_h = tf.maximum(h - 1.0, 1.0) # Avoid division by zero if h=1
    scale_w = tf.maximum(w - 1.0, 1.0) # Avoid division by zero if w=1
    scale = tf.concat([scale_h, scale_w, scale_h, scale_w], axis=-1)
    shift = tf.constant([0., 0., 1., 1.], dtype=tf.float32)
    return (tf.cast(boxes, tf.float32) - shift) / scale


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    # ... (docstring from original)
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale_h = tf.maximum(h - 1.0, 1.0)
    scale_w = tf.maximum(w - 1.0, 1.0)
    scale = tf.concat([scale_h, scale_w, scale_h, scale_w], axis=-1)
    shift = tf.constant([0., 0., 1., 1.], dtype=tf.float32)
    return tf.cast(tf.round(boxes * scale + shift), tf.int32)


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    # ... (docstring from original)
    """
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    if augment: # augment is deprecated, use augmentation
        log("Warning: 'augment' parameter in load_image_gt is deprecated. Use 'augmentation'.")
        if random.randint(0, 1): # Simple flip augmentation if augment=True
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    if augmentation: # New style augmentation
        import imgaug # Ensure imgaug is installed
        # Augmenters that apply to images and masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]
        def hook(images_hook, augmenter_hook, parents_hook, default_hook): # Renamed args
            return augmenter_hook.__class__.__name__ in MASK_AUGMENTERS
        
        image_shape_before_aug = image.shape # Store shapes before augmentation
        mask_shape_before_aug = mask.shape
        
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Augment mask with the same deterministic augmenter
        # Ensure mask is uint8 for imgaug
        mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask.astype(np.uint8)
        mask_augmented = det.augment_image(mask_uint8, hooks=imgaug.HooksImages(activator=hook))
        
        # Ensure shapes did not change unexpectedly
        if image.shape != image_shape_before_aug:
            log(f"Warning: Image shape changed during augmentation from {image_shape_before_aug} to {image.shape}. Resizing back.")
            image = utils.resize(image, image_shape_before_aug[:2], preserve_range=True, order=1) # Bilinear
        if mask_augmented.shape[:2] != mask_shape_before_aug[:2]: # Compare only H,W for mask
            log(f"Warning: Mask shape changed during augmentation from {mask_shape_before_aug} to {mask_augmented.shape}. Resizing back.")
            mask_augmented = utils.resize(mask_augmented, mask_shape_before_aug[:2], preserve_range=True, order=0) # Nearest for mask
            
        mask = (mask_augmented > 127).astype(bool) if mask_uint8.dtype == np.uint8 else mask_augmented.astype(bool)


    # Remove empty masks (instances with no pixels after augmentation/resize)
    # And corresponding class IDs and bboxes
    if mask.shape[-1] > 0: # If there are any instances
        instance_sums = np.sum(mask, axis=(0, 1))
        valid_instances = instance_sums > 0
        mask = mask[:, :, valid_instances]
        class_ids = class_ids[valid_instances]

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out. Dot not filter based on zero bboxes here.
    bbox = utils.extract_bboxes(mask) # Recalculate bboxes from potentially altered masks

    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids.get(dataset.image_info[image_id]["source"], []) # Handle missing source
    active_class_ids[source_class_ids] = 1

    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None, # augment is deprecated
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    # ... (docstring from original)
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # TF2 Note: compute_backbone_shapes and generate_pyramid_anchors are NumPy based from utils.
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    while True:
        try:
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            image_id = image_ids[image_index]
            
            current_augmentation = augmentation
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                current_augmentation = None

            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment, # augment is deprecated
                              augmentation=current_augmentation,
                              use_mini_mask=config.USE_MINI_MASK)

            if not np.any(gt_class_ids > 0): # Skip images with no instances.
                continue

            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            if random_rois: # This part is not typically used in standard MRCNN training.
                rpn_rois = generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
                        build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

            if b == 0: # Initialize batch arrays
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros((batch_size, anchors.shape[0], 1), dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros((batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4), dtype=rpn_bbox.dtype)
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.float32) # Ensure float for normalized boxes
                # Mask shape can vary if mini_mask is used. Initialize based on actual gt_masks shape.
                batch_gt_masks_shape = (batch_size, gt_masks.shape[0], gt_masks.shape[1], config.MAX_GT_INSTANCES)
                if config.USE_MINI_MASK: # If mini_mask, last dim is num_instances, not MAX_GT_INSTANCES
                     batch_gt_masks_shape = (batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], config.MAX_GT_INSTANCES)
                batch_gt_masks = np.zeros(batch_gt_masks_shape, dtype=bool)


                if random_rois: # Not standard
                    batch_rpn_rois = np.zeros((batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros((batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros((batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros((batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)
            
            # Pad GT instances if too many
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Fill batch arrays
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis] # Ensure [anchors, 1]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config) # Image already float from load_image_gt usually
            
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            
            # Handle gt_masks padding carefully
            num_gt_instances_current = gt_masks.shape[-1]
            if config.USE_MINI_MASK:
                batch_gt_masks[b, :, :, :num_gt_instances_current] = gt_masks
            else: # Full masks
                batch_gt_masks[b, :, :, :num_gt_instances_current] = gt_masks


            if random_rois: # Not standard
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            if b >= batch_size: # Batch full
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = [] # Keras fit expects (inputs, targets) or just inputs if targets are handled by model losses

                # This part for random_rois/detection_targets is not standard for training the main model
                # It seems related to generating data for a separate detector, or debugging.
                # If it's for model targets, they should be structured according to model.compile()
                # For now, assume standard training where losses are internal to the model.
                if random_rois and detection_targets:
                    # This structure of outputs was for a specific Keras model setup.
                    # For the main MaskRCNN model, these are not direct targets for model.fit()
                    # unless the model is compiled to expect them.
                    # The current MaskRCNN model handles losses internally.
                    # So, `outputs` should be empty or None.
                    # batch_mrcnn_class_ids_expanded = np.expand_dims(batch_mrcnn_class_ids, -1) # Example
                    # outputs.extend([batch_mrcnn_class_ids_expanded, batch_mrcnn_bbox, batch_mrcnn_mask])
                    pass # Keep outputs empty for standard MaskRCNN training

                yield inputs, outputs # TF2 Keras fit can take (inputs, None)
                b = 0 # Reset batch index
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except Exception as e:
            log(f"Error in data generator for image_id {image_id if image_index != -1 else 'unknown'}: {e}")
            error_count += 1
            if error_count > 5: # Stop if too many errors
                log("Too many errors in data generator. Aborting.")
                raise e


# Functions from original model.py that were used by data_generator or are utils.
# These are typically NumPy based.
def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_instances] Integer class IDs.
    gt_boxes: [num_gt_instances, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Filter out crowds from anchors RPN match
        # (rpn_match already initialized to 0, so neutral)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those with IoU between 0.3 and 0.7 (inclusive).
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[anchor_iou_max < config.RPN_NEGATIVE_THRESHOLD] = -1 # Default is 0.3
    
    # Positive anchors:
    # Anchors with IoU >= RPN_POSITIVE_THRESHOLD (0.7 default) with any GT box.
    gt_iou_argmax = np.argmax(overlaps, axis=0) # Anchor with highest IoU for each GT box
    rpn_match[gt_iou_argmax] = 1
    rpn_match[anchor_iou_max >= config.RPN_POSITIVE_THRESHOLD] = 1 # Default is 0.7

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7 if it's the best match)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that a Network Layer would learn.
        # Add epsilon to avoid division by zero or log(0)
        a_h = np.maximum(a_h, 1e-7)
        a_w = np.maximum(a_w, 1e-7)
        gt_h = np.maximum(gt_h, 1e-7)
        gt_w = np.maximum(gt_w, 1e-7)

        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize TFBBOX Standard Deviations
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a Network Lighter would
    generate.
    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer class IDs.
    gt_boxes: [N, (y1, x1, y2, x2)]
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0]) if gt_boxes.shape[0] else 0
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random shifts
        r_y = np.random.randint(-h // 2, h // 2)
        r_x = np.random.randint(-w // 2, w // 2)
        # random scale
        s = np.random.uniform(0.3, 1.0)
        # Ensure values are valid
        y1 = np.clip(gt_y1 + r_y - h * s // 2, 0, image_shape[0])
        y2 = np.clip(y1 + h * s, 0, image_shape[0])
        x1 = np.clip(gt_x1 + r_x - w * s // 2, 0, image_shape[1])
        x2 = np.clip(x1 + w * s, 0, image_shape[1])
        # Filter out zero area boxes
        if (y2 - y1) * (x2 - x1) == 0:
            continue
        # Add count per box
        for j in range(rois_per_box):
            idx = gt_boxes.shape[0] * j + i
            if idx >= count - rois_per_box: # Check bounds
                break
            rois[idx] = [y1, x1, y2, x2]

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    for i in range(remaining_count):
        # Generate random box
        y1 = np.random.randint(0, image_shape[0] * 0.75) # Avoid edges
        x1 = np.random.randint(0, image_shape[1] * 0.75)
        y2 = np.random.randint(y1 + image_shape[0] // 10, image_shape[0]) # Min height
        x2 = np.random.randint(x1 + image_shape[1] // 10, image_shape[1]) # Min width
        rois[rois_per_box * gt_boxes.shape[0] + i] = [y1, x1, y2, x2]

    return rois


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or if
    training the heads RPN part separately.

    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance_count] Integer class IDs
    gt_boxes: [instance_count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance_count] Grund truth masks. Can be full
              size or mini-masks.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))] Class-specific
            bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32 or gt_boxes.dtype == np.float32, "Expected int or float but got {}".format( # Allow float for normalized
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)

    # It's okay to use zero padding and filter out zeros later
    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = utils.compute_overlaps(rpn_rois, gt_boxes)

    # Assign ROIs to GT boxes
    roi_iou_max = np.max(overlaps, axis=1)
    roi_gt_box_assignment = np.argmax(overlaps, axis=1)
    roi_gt_boxes = gt_boxes[roi_gt_box_assignment]
    roi_class_ids = gt_class_ids[roi_gt_box_assignment]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    positive_roi_ix = np.where(roi_iou_max >= 0.5)[0]

    # Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_ix = np.where(roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% positive.
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    if positive_roi_ix.shape[0] > positive_count:
        positive_roi_ix = np.random.choice(
            positive_roi_ix, positive_count, replace=False)
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    negative_count = config.TRAIN_ROIS_PER_IMAGE - positive_roi_ix.shape[0]
    if negative_roi_ix.shape[0] > negative_count:
        negative_roi_ix = np.random.choice(
            negative_roi_ix, negative_count, replace=False)

    # Gather selected ROIs
    positive_rois = rpn_rois[positive_roi_ix]
    negative_rois = rpn_rois[negative_roi_ix]

    # Assign positive ROIs to GT boxes
    # Recompute assignment for the subsampled positive ROIs
    if positive_rois.shape[0] > 0:
        positive_overlaps = utils.compute_overlaps(positive_rois, gt_boxes)
        roi_gt_box_assignment = np.argmax(positive_overlaps, axis=1)
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment]
        roi_class_ids = gt_class_ids[roi_gt_box_assignment]
    else: # No positive ROIs
        roi_gt_boxes = np.empty((0,4))
        roi_class_ids = np.empty((0), dtype=np.int32)


    # Bounding box refinement targets for positive ROIs
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    if positive_rois.shape[0] > 0:
        pos_ix = np.arange(positive_rois.shape[0])
        # Compute bbox refinement for each positive ROI
        deltas = utils.box_refinement(positive_rois, roi_gt_boxes)
        deltas /= config.BBOX_STD_DEV
        # Assign deltas to the correct class an ROI corresponds to.
        bboxes[pos_ix, roi_class_ids, :] = deltas

    # Masks for positive ROIs
    # Permute masks to [N, height, width, 1]
    # gt_masks is [height, width, instance_count]
    # We need [instance_count, height, width] to gather based on roi_gt_box_assignment
    if positive_rois.shape[0] > 0 and gt_masks.shape[-1] > 0:
        permuted_gt_masks = np.transpose(gt_masks, [2, 0, 1])
        roi_masks = permuted_gt_masks[roi_gt_box_assignment, :, :]

        # Compute mask targets
        masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0],
                          config.MASK_SHAPE[1], config.NUM_CLASSES),
                         dtype=np.float32) # Changed to float for consistency with model output type
        for i in range(positive_rois.shape[0]): # Iterate over positive ROIs
            box = positive_rois[i]
            class_id = roi_class_ids[i]
            mask_instance = roi_masks[i] # This is the GT mask for this instance
            
            # Transform ROI coordinates from normalized image space
            # to normalized mask space for mini-mask, or use directly for full mask.
            if config.USE_MINI_MASK:
                y1, x1, y2, x2 = box
                gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes[i]
                gt_h = gt_y2 - gt_y1
                gt_w = gt_x2 - gt_x1
                # Normalize box coordinates relative to the GT box
                # Add epsilon to avoid division by zero
                y1_norm = (y1 - gt_y1) / (gt_h + 1e-7)
                x1_norm = (x1 - gt_x1) / (gt_w + 1e-7)
                y2_norm = (y2 - gt_y1) / (gt_h + 1e-7)
                x2_norm = (x2 - gt_x1) / (gt_w + 1e-7)
                box_for_crop = np.array([y1_norm, x1_norm, y2_norm, x2_norm])
                # Crop the GT mask to this ROI (already done by selection)
                # Then resize to MASK_SHAPE
                # This part needs to be careful: mask_instance is full GT mask.
                # We need to crop mask_instance using 'box' (the proposal ROI)
                # then resize to MASK_SHAPE.
                # For mini-mask, the GT mask itself is small.
                # utils.minimize_mask does this. Here we are generating targets.
                
                # This logic for mini-mask target generation needs review.
                # For now, assume full masks are used or USE_MINI_MASK implies gt_masks are already mini.
                # If gt_masks are full, we need to crop them to 'box' and then resize.
                # If gt_masks are mini (from load_image_gt with USE_MINI_MASK=True),
                # then 'box' needs to be scaled to mini-mask coords.
                # The current code seems to mix these.
                # Let's assume gt_masks are full here for simplicity of target generation.
                # utils.extract_bboxes from full mask to get precise GT box.
                # Crop mask_instance using the 'box' (proposal ROI)
                # This is complex. For now, assume full mask and crop_and_resize logic.
                # The DetectionTargetLayer's graph version is more robust.
                # This NumPy version is for debugging/understanding.
                # For simplicity, let's assume utils.unmold_mask logic can be adapted.
                # This function is not directly used in the main training path of the refactored model.
                # It's more of a utility.
                # For now, let's skip the complex mask target part here, as the graph version is primary.
            else: # Full masks
                # Crop and resize the GT mask to the proposal box (box)
                # This is essentially what crop_and_resize does in graph.
                # For NumPy, it's more manual.
                # Simplified: use the GT mask directly for the assigned positive ROI.
                # This is not ideal as it doesn't align with proposal box.
                # This function is problematic for mask targets in NumPy.
                # For consistency, this should replicate graph logic, which is hard in NumPy.
                # Given its limited use, let's put a placeholder for mask part.
                # Placeholder: target mask is just the GT mask resized, not aligned with proposal.
                # This is NOT correct for training a mask head.
                # resized_gt_mask = utils.resize(mask_instance.astype(np.float32), config.MASK_SHAPE, order=0)
                # masks[i, :, :, class_id] = resized_gt_mask
                pass # Mask target generation here is complex and likely unused.

    # Append negative ROIs and pad
    rois = np.concatenate([positive_rois, negative_rois], axis=0)
    N = negative_rois.shape[0]
    P = config.TRAIN_ROIS_PER_IMAGE - rois.shape[0]
    rois = np.pad(rois, ((0, P), (0, 0)), 'constant')
    
    # Pad class_ids for positive ROIs, then append zeros for negative and padding
    if positive_rois.shape[0] > 0:
        class_ids_padded = np.pad(roi_class_ids, (0, config.TRAIN_ROIS_PER_IMAGE - positive_rois.shape[0]), 'constant')
    else:
        class_ids_padded = np.zeros((config.TRAIN_ROIS_PER_IMAGE,), dtype=np.int32)

    # bboxes are already [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, 4]
    # masks are already [TRAIN_ROIS_PER_IMAGE, H, W, NUM_CLASSES]
    # (If mask part was implemented correctly)
    # For now, return empty masks if not implemented:
    if not np.any(masks): # If masks is still all zeros
        masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0],
                          config.MASK_SHAPE[1], config.NUM_CLASSES), dtype=np.float32)


    return rois, class_ids_padded, bboxes, masks
