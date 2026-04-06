"""Mask R-CNN backend using the vendored mrcnn/ module.

This backend wraps the Matterport Mask R-CNN implementation that ships
with the repository.  It requires TensorFlow 2.x.

.. deprecated::
    The Mask R-CNN backend is deprecated and will be removed in a future
    version.  Use ``--backend yolov8`` instead.
"""

import warnings

import numpy as np

from backends import DetectionBackend, DetectionResult


class MaskRCNNBackend(DetectionBackend):
    """Detection backend using the vendored Matterport Mask R-CNN."""

    def load_model(self, weights_path, device=None, **kwargs):
        """Load a Mask R-CNN model.

        Parameters
        ----------
        weights_path : str
            Path to an ``.h5`` weights file.
        device : str or None
            ``"cpu"`` or ``"gpu"``.
        **kwargs
            ``num_classes`` (int): Total number of classes including
            background.  Required.
            ``min_confidence`` (float): Override
            ``DETECTION_MIN_CONFIDENCE``.
            ``command`` (str): ``"annotateCoco"`` or
            ``"annotateCustom"`` — controls the config used.
        """
        import tensorflow as tf
        from mrcnn.config import Config
        from mrcnn.model import MaskRCNN

        warnings.warn(
            "The Mask R-CNN backend is deprecated and will be removed "
            "in a future release.  Please migrate to --backend yolov8.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Device selection
        if device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        elif device == "gpu":
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise RuntimeError(
                    "No GPU devices available. Use --device cpu or omit --device."
                )

        num_classes = kwargs.get("num_classes", 81)
        min_confidence = kwargs.get("min_confidence")
        config_name = kwargs.get("command", "annotateCoco")

        class _InferenceConfig(Config):
            NAME = "inferenceCoco" if config_name == "annotateCoco" else "inferenceCustom"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = num_classes

        config = _InferenceConfig()
        if min_confidence is not None:
            config.DETECTION_MIN_CONFIDENCE = min_confidence
        config.display()

        model = MaskRCNN(mode="inference", config=config, model_dir="./")
        model.load_weights(weights_path, by_name=True)
        return model

    def detect(self, model, image, confidence_threshold=0.7):
        """Run Mask R-CNN inference on a single image.

        Returns a list of :class:`DetectionResult` instances.
        """
        results = model.detect([image], verbose=0)
        result = results[0]
        detections = []
        n = len(result["class_ids"])
        for i in range(n):
            score = float(result["scores"][i])
            if score < confidence_threshold:
                continue
            # Mask R-CNN rois: (y1, x1, y2, x2)
            y1, x1, y2, x2 = result["rois"][i]
            bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            mask = result["masks"][:, :, i] if result["masks"] is not None else None
            detections.append(DetectionResult(
                class_id=int(result["class_ids"][i]),
                label="",  # filled by caller using class_names
                bbox=bbox,
                score=score,
                mask=mask,
            ))
        return detections

    def get_class_names(self, model):
        """Mask R-CNN does not embed class names in the model.

        The caller must supply them explicitly (e.g. COCO_DATASET_LABELS).
        This method returns an empty list as a sentinel so the caller
        knows to provide its own names.
        """
        return []
