"""YOLOv8 backend using the Ultralytics package.

This backend provides object detection and instance segmentation using
YOLOv8 models.  It supports both built-in COCO pretrained models and
user-supplied ``.pt`` weight files for custom-trained models.
"""

import numpy as np

from backends import DetectionBackend, DetectionResult


# COCO class names in the order used by Ultralytics YOLOv8.
# This is the standard 80-class COCO ordering (no background class).
YOLOV8_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Map from model size short-name to Ultralytics model identifier.
_MODEL_SIZE_MAP = {
    "nano": "n",
    "small": "s",
    "medium": "m",
    "large": "l",
    "xlarge": "x",
}


def _build_model_name(model_size="medium", segmentation=True):
    """Build the Ultralytics model file name.

    Parameters
    ----------
    model_size : str
        One of ``"nano"``, ``"small"``, ``"medium"``, ``"large"``,
        ``"xlarge"``.
    segmentation : bool
        If ``True``, use the segmentation variant (``-seg``).

    Returns
    -------
    str
        E.g. ``"yolov8m-seg.pt"`` or ``"yolov8m.pt"``.
    """
    suffix = _MODEL_SIZE_MAP.get(model_size, model_size)
    seg = "-seg" if segmentation else ""
    return "yolov8{}{}.pt".format(suffix, seg)


class YOLOv8Backend(DetectionBackend):
    """Detection backend using Ultralytics YOLOv8."""

    def load_model(self, weights_path, device=None, **kwargs):
        """Load a YOLOv8 model.

        Parameters
        ----------
        weights_path : str
            Path to a ``.pt`` weights file **or** a model identifier
            such as ``"yolov8m-seg.pt"`` (Ultralytics downloads it
            automatically).
        device : str or None
            ``"cpu"``, ``"gpu"``, or ``None``.
        **kwargs
            ``model_size`` (str): Convenience shortcut.  If
            *weights_path* equals the literal ``"coco_weights"`` or
            ``"coco"``, this is used to build the model name.
            ``segmentation`` (bool): Use the segmentation variant
            (default ``True``).
        """
        from ultralytics import YOLO

        model_size = kwargs.get("model_size", "medium")
        segmentation = kwargs.get("segmentation", True)

        # If the user passes the legacy "coco_weights" sentinel or
        # "coco", build the standard Ultralytics model name.
        if weights_path.lower() in ("coco_weights", "coco"):
            weights_path = _build_model_name(model_size, segmentation)

        model = YOLO(weights_path)

        # Store device preference on the model object.
        if device == "gpu":
            model._device = "0"  # CUDA device 0
        elif device == "cpu":
            model._device = "cpu"
        else:
            model._device = None  # auto
        return model

    def detect(self, model, image, confidence_threshold=0.7):
        """Run YOLOv8 inference on a single image.

        Returns a list of :class:`DetectionResult` instances.
        """
        device = getattr(model, "_device", None)
        predict_kwargs = {
            "conf": confidence_threshold,
            "verbose": False,
        }
        if device is not None:
            predict_kwargs["device"] = device

        results = model.predict(image, **predict_kwargs)
        result = results[0]

        detections = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract masks if the model supports segmentation.
        masks_data = None
        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()

        class_names = self.get_class_names(model)

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())

            # YOLOv8 boxes are in xyxy format.
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))

            # Build full-resolution binary mask if available.
            mask = None
            if masks_data is not None and i < len(masks_data):
                raw = masks_data[i]
                # Masks from Ultralytics may have a different spatial
                # resolution than the original image.  Resize if needed.
                h_img, w_img = image.shape[:2]
                if raw.shape[0] != h_img or raw.shape[1] != w_img:
                    import cv2
                    raw = cv2.resize(
                        raw.astype(np.float32), (w_img, h_img),
                        interpolation=cv2.INTER_LINEAR,
                    )
                mask = (raw > 0.5).astype(bool)

            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            detections.append(DetectionResult(
                class_id=cls_id,
                label=label,
                bbox=bbox,
                score=score,
                mask=mask,
            ))
        return detections

    def get_class_names(self, model):
        """Return class names from the loaded model.

        Ultralytics models store class names in ``model.names`` as a
        dict ``{0: 'person', 1: 'bicycle', ...}``.
        """
        names = getattr(model, "names", None)
        if names is None:
            return list(YOLOV8_COCO_CLASSES)
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        return list(names)
