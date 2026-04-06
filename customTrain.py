import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import warnings
import tempfile

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "./mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Mask R-CNN Configurations & Dataset (legacy backend)
############################################################


def _get_mrcnn_modules():
    """Lazy-import Mask R-CNN modules (requires TensorFlow)."""
    from mrcnn.config import Config
    from mrcnn import model as modellib, utils
    return Config, modellib, utils


class CustomConfig:
    """Configuration for training on a custom single-class dataset.

    This is instantiated only when ``--backend maskrcnn`` is selected.
    The actual base class (:class:`mrcnn.config.Config`) is resolved
    at runtime to avoid importing TensorFlow at module level.
    """

    @staticmethod
    def create():
        Config, _, _ = _get_mrcnn_modules()

        class _CustomConfig(Config):
            NAME = "CustomLabel"
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + 1  # Background + customLabel
            STEPS_PER_EPOCH = 100
            DETECTION_MIN_CONFIDENCE = 0.9

        return _CustomConfig()


class CustomDataset:
    """Wrapper around the Mask R-CNN Dataset class for VIA JSON data."""

    @staticmethod
    def create():
        _, _, utils_mod = _get_mrcnn_modules()

        class _CustomDataset(utils_mod.Dataset):

            def load_custom(self, dataset_dir, subset):
                """Load a subset of the Custom dataset.
                dataset_dir: Root directory of the dataset.
                subset: Subset to load: train or val
                """
                self.add_class("CustomLabel", 1, "CustomLabel")

                assert subset in ["train", "val"]
                dataset_dir = os.path.join(dataset_dir, subset)

                annotations = json.load(
                    open(os.path.join(dataset_dir, "via_region_data.json")))
                annotations = list(annotations.values())
                annotations = [a for a in annotations if a['regions']]

                for a in annotations:
                    if type(a['regions']) is dict:
                        polygons = [r['shape_attributes']
                                    for r in a['regions'].values()]
                    else:
                        polygons = [r['shape_attributes']
                                    for r in a['regions']]

                    image_path = os.path.join(dataset_dir, a['filename'])
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]

                    self.add_image(
                        "customLabel",
                        image_id=a['filename'],
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons)

            def load_mask(self, image_id):
                image_info = self.image_info[image_id]
                if image_info["source"] != "customLabel":
                    return super(self.__class__, self).load_mask(image_id)

                info = self.image_info[image_id]
                mask = np.zeros(
                    [info["height"], info["width"], len(info["polygons"])],
                    dtype=np.uint8)
                for i, p in enumerate(info["polygons"]):
                    rr, cc = skimage.draw.polygon(
                        p['all_points_y'], p['all_points_x'])
                    mask[rr, cc, i] = 1
                return mask.astype(bool), np.ones(
                    [mask.shape[-1]], dtype=np.int32)

            def image_reference(self, image_id):
                info = self.image_info[image_id]
                if info["source"] == "customLabel":
                    return info["path"]
                return super(self.__class__, self).image_reference(image_id)

        return _CustomDataset()


def train_maskrcnn(args):
    """Train using the Mask R-CNN backend (legacy)."""
    warnings.warn(
        "The Mask R-CNN backend is deprecated and will be removed "
        "in a future release.  Please migrate to --backend yolov8.",
        DeprecationWarning,
        stacklevel=2,
    )

    _, modellib, utils_mod = _get_mrcnn_modules()

    config = CustomConfig.create()

    model = modellib.MaskRCNN(
        mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils_mod.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Training dataset.
    dataset_train = CustomDataset.create()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset.create()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args.epochs,
                layers='heads')


############################################################
#  YOLOv8 Training
############################################################


def _generate_dataset_yaml(dataset_dir, label_name, output_path=None):
    """Create a YAML dataset config for Ultralytics YOLOv8 training.

    Expects the dataset directory to contain ``train/`` and ``val/``
    sub-directories, each with ``images/`` and ``labels/`` folders in
    standard YOLO format.

    Parameters
    ----------
    dataset_dir : str
        Root of the dataset.
    label_name : str
        Name of the single custom class.
    output_path : str or None
        Where to write the YAML file.  If ``None``, a temporary file
        is created.

    Returns
    -------
    str
        Path to the generated YAML file.
    """
    import yaml

    data = {
        "path": os.path.abspath(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {0: label_name},
    }
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".yaml", prefix="yolo_data_")
        os.close(fd)
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print("Generated dataset config: {}".format(output_path))
    return output_path


def convert_via_to_yolo(via_json_path, output_dir):
    """Convert VIA JSON annotations to YOLO .txt label files.

    Each image's annotations are written to a ``.txt`` file alongside
    the images.  The output follows the YOLO format:
    ``class_id cx cy w h`` (normalized 0-1).

    Parameters
    ----------
    via_json_path : str
        Path to the ``via_region_data.json`` file.
    output_dir : str
        Directory to write the ``.txt`` label files.
    """
    with open(via_json_path, 'r') as f:
        annotations = json.load(f)
    annotations = list(annotations.values())
    annotations = [a for a in annotations if a.get('regions')]

    os.makedirs(output_dir, exist_ok=True)

    for a in annotations:
        filename = a['filename']
        base = os.path.splitext(filename)[0]
        image_path = os.path.join(os.path.dirname(via_json_path), filename)

        # Read image dimensions
        image = skimage.io.imread(image_path)
        img_h, img_w = image.shape[:2]

        regions = a['regions']
        if isinstance(regions, dict):
            regions = list(regions.values())

        lines = []
        for region in regions:
            shape = region['shape_attributes']
            xs = shape['all_points_x']
            ys = shape['all_points_y']
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = x_max - x_min
            h = y_max - y_min
            cx = (x_min + w / 2.0) / img_w
            cy = (y_min + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(cx, cy, nw, nh))

        txt_path = os.path.join(output_dir, base + '.txt')
        with open(txt_path, 'w') as f:
            f.write("\n".join(lines))

    print("Converted {} images to YOLO format in {}".format(
        len(annotations), output_dir))


def train_yolov8(args):
    """Train using the YOLOv8 backend."""
    from ultralytics import YOLO

    # Build the base model name from model_size.
    from backends.yolov8 import _build_model_name
    if args.weights.lower() in ("coco", "coco_weights"):
        model_name = _build_model_name(args.model_size, segmentation=False)
    else:
        model_name = args.weights

    model = YOLO(model_name)

    # Generate dataset YAML config
    label_name = args.label if args.label else "CustomLabel"
    data_yaml = _generate_dataset_yaml(args.dataset, label_name)

    print("Training YOLOv8 model: {}".format(model_name))
    print("Dataset config: {}".format(data_yaml))
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=640,
        project=args.logs,
        name="yolov8_custom",
    )
    print("Training complete. Results saved to: {}".format(args.logs))


############################################################
#  Training (entry point)
############################################################

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train a detection model on custom objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'convert'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the Custom dataset')
    parser.add_argument('--weights', required=False,
                        default="coco",
                        metavar="/path/to/weights",
                        help="Path to weights file or 'coco' (default: coco)")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--backend', default='yolov8',
                        choices=['maskrcnn', 'yolov8'],
                        help='Training backend (default: yolov8)')
    parser.add_argument('--model_size', default='medium',
                        choices=['nano', 'small', 'medium', 'large', 'xlarge'],
                        help='YOLOv8 model size (default: medium). '
                             'Ignored when --backend=maskrcnn.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--label', default='CustomLabel',
                        help='Name of the custom class label (default: CustomLabel)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        if not args.dataset:
            parser.error("Argument --dataset is required for training")
    elif args.command == "convert":
        if not args.dataset:
            parser.error("Argument --dataset is required for conversion")
    else:
        parser.error("'{}' is not recognized. Use 'train' or 'convert'".format(
            args.command))

    print("Backend: ", args.backend)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    if args.command == "train":
        if args.backend == "maskrcnn":
            train_maskrcnn(args)
        else:
            train_yolov8(args)

    elif args.command == "convert":
        # Convert VIA JSON annotations to YOLO format.
        for subset in ["train", "val"]:
            via_path = os.path.join(args.dataset, subset, "via_region_data.json")
            if os.path.exists(via_path):
                out_dir = os.path.join(args.dataset, subset, "labels")
                convert_via_to_yolo(via_path, out_dir)
            else:
                print("Skipping {}: {} not found".format(subset, via_path))
