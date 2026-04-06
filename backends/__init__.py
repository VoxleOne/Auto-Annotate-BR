"""Backend abstraction layer for object detection models.

Provides a unified interface so that different detection backends
(Mask R-CNN, YOLOv8, etc.) can be used interchangeably by the
annotation pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DetectionResult:
    """Standardized detection result returned by all backends.

    Attributes:
        class_id: Integer class ID (0-based, no background class).
        label: Human-readable class label string.
        bbox: Bounding box as ``(x, y, width, height)`` in pixel
            coordinates (top-left origin, absolute values).
        score: Detection confidence score in ``[0, 1]``.
        mask: Optional binary mask of shape ``(H, W)`` where *H* and *W*
            match the original image dimensions.  May be ``None`` when
            the backend does not produce masks.
    """
    class_id: int
    label: str
    bbox: tuple  # (x, y, w, h)
    score: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)


class DetectionBackend(ABC):
    """Abstract base class for detection backends."""

    @abstractmethod
    def load_model(self, weights_path, device=None, **kwargs):
        """Load a detection model.

        Parameters
        ----------
        weights_path : str
            Path to weights file, or a model identifier understood by
            the concrete backend (e.g. ``"yolov8m-seg.pt"``).
        device : str or None
            ``"cpu"`` or ``"gpu"``.  ``None`` lets the backend choose.
        **kwargs
            Backend-specific options (e.g. ``num_classes``,
            ``min_confidence``).

        Returns
        -------
        object
            The loaded model object (backend-specific).
        """

    @abstractmethod
    def detect(self, model, image, confidence_threshold=0.7):
        """Run inference on a single image.

        Parameters
        ----------
        model : object
            Model returned by :meth:`load_model`.
        image : numpy.ndarray
            Image array with shape ``(H, W, 3)`` in RGB uint8 format.
        confidence_threshold : float
            Minimum confidence to keep a detection.

        Returns
        -------
        list[DetectionResult]
            Detections found in the image.
        """

    @abstractmethod
    def get_class_names(self, model):
        """Return ordered list of class names the model can detect.

        The list should **not** include a background class.  Index *i*
        in the returned list corresponds to ``class_id=i`` in
        :class:`DetectionResult`.

        Parameters
        ----------
        model : object
            Model returned by :meth:`load_model`.

        Returns
        -------
        list[str]
        """


def get_backend(name):
    """Return a backend instance by name.

    Parameters
    ----------
    name : str
        ``"maskrcnn"`` or ``"yolov8"``.

    Returns
    -------
    DetectionBackend

    Raises
    ------
    ValueError
        If *name* is not recognized.
    ImportError
        If required dependencies for the backend are not installed.
    """
    name = name.lower()
    if name == "maskrcnn":
        from backends.maskrcnn import MaskRCNNBackend
        return MaskRCNNBackend()
    if name == "yolov8":
        from backends.yolov8 import YOLOv8Backend
        return YOLOv8Backend()
    raise ValueError(
        "Unknown backend '{}'. Choose from: maskrcnn, yolov8".format(name)
    )
