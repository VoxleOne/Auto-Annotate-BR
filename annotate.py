import numpy as np
from shapely.geometry import Polygon, MultiPolygon 
from skimage import measure   
import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from mrcnn import utils
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def annotateResult(result, image_name, labels, class_names):
    """Annotate a single result for one or more labels."""
    n = len(result['class_ids'])
    annotations = []
    annotationId = 1
    for i in range(n):
        detected_label = class_names[result['class_ids'][i]]
        if detected_label in labels:
            annotation = create_sub_mask_annotation(result['masks'][:, :, i], result['rois'][i],
                                                    annotationId, result['class_ids'][i], image_name,
                                                    class_names)
            annotations.append(annotation)
            annotationId += 1
    return annotations


def create_sub_mask_annotation(sub_mask, bounding_box, annotationId, classId, image_name, class_names):
    # Find contours (boundary lines) around each sub-mask
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    # area = multi_poly.area

    annotation = {
        'filename': image_name,
        'id': annotationId,
        'label': str(class_names[classId]),
        'bbox': bbox,
        'segmentation': segmentations,
    }

    return annotation



def writeToJSONFile(path, fileName, data, overwrite=True):
    fileName = fileName.split(".")[0]
    filePathNameWExt = os.path.join(path, fileName + '.json')
    if not overwrite and os.path.exists(filePathNameWExt):
        print("Skipping (file exists): " + filePathNameWExt)
        return
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)


def writeCocoJSON(path, fileName, annotations, img_width, img_height):
    """Write annotations in COCO JSON format."""
    fileName = fileName.split(".")[0]
    filePathNameWExt = os.path.join(path, fileName + '_coco.json')
    coco = {
        "images": [{
            "id": 1,
            "file_name": fileName,
            "width": img_width,
            "height": img_height,
        }],
        "annotations": [],
        "categories": []
    }
    category_ids = {}
    for ann in annotations:
        label = ann['label']
        if label not in category_ids:
            cat_id = len(category_ids) + 1
            category_ids[label] = cat_id
            coco["categories"].append({"id": cat_id, "name": label})
        coco_ann = {
            "id": ann['id'],
            "image_id": 1,
            "category_id": category_ids[label],
            "bbox": list(ann['bbox']),
            "segmentation": ann['segmentation'],
            "area": ann['bbox'][2] * ann['bbox'][3],
            "iscrowd": 0,
        }
        coco["annotations"].append(coco_ann)
    with open(filePathNameWExt, 'w') as fp:
        json.dump(coco, fp)


def writeVocXML(path, fileName, annotations, img_width, img_height):
    """Write annotations in Pascal VOC XML format."""
    baseName = fileName.split(".")[0]
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = fileName
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"

    for ann in annotations:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = ann['label']
        bndbox = ET.SubElement(obj, "bndbox")
        x, y, w, h = ann['bbox']
        ET.SubElement(bndbox, "xmin").text = str(int(x))
        ET.SubElement(bndbox, "ymin").text = str(int(y))
        ET.SubElement(bndbox, "xmax").text = str(int(x + w))
        ET.SubElement(bndbox, "ymax").text = str(int(y + h))

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    filePathNameWExt = os.path.join(path, baseName + '.xml')
    with open(filePathNameWExt, 'w') as fp:
        fp.write(xml_str)


def writeYoloTxt(path, fileName, annotations, img_width, img_height, label_to_id):
    """Write annotations in YOLO .txt format (class_id cx cy w h, normalized)."""
    baseName = fileName.split(".")[0]
    filePathNameWExt = os.path.join(path, baseName + '.txt')
    lines = []
    for ann in annotations:
        class_id = label_to_id.get(ann['label'], 0)
        x, y, w, h = ann['bbox']
        cx = (x + w / 2.0) / img_width
        cy = (y + h / 2.0) / img_height
        nw = w / img_width
        nh = h / img_height
        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(class_id, cx, cy, nw, nh))
    with open(filePathNameWExt, 'w') as fp:
        fp.write("\n".join(lines))


def annotateAndSaveAnnotations(r, directory, image_name, labels, class_names,
                               overwrite=True, output_format="auto-annotate",
                               img_width=0, img_height=0, label_to_id=None):
    annotationsJson = annotateResult(r, image_name, labels, class_names)
    if output_format == "coco":
        writeCocoJSON(directory, image_name, annotationsJson, img_width, img_height)
    elif output_format == "voc":
        writeVocXML(directory, image_name, annotationsJson, img_width, img_height)
    elif output_format == "yolo":
        writeYoloTxt(directory, image_name, annotationsJson, img_width, img_height,
                     label_to_id or {})
    else:
        writeToJSONFile(directory, image_name, annotationsJson, overwrite=overwrite)


def annotateImagesInDirectory(rcnn, directory_path, labels, class_names,
                              display_masked=False, overwrite=True,
                              output_format="auto-annotate"):
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    image_files = [
        f for f in sorted(os.listdir(directory_path))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ]

    # Build label-to-id mapping for YOLO format
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    iterator = tqdm(image_files, desc="Annotating") if has_tqdm else image_files
    for fileName in iterator:
        try:
            # load image
            print("Evaluating Image: " + fileName)
            img = load_img(os.path.join(directory_path, fileName))
            img = img_to_array(img)
            img_height, img_width = img.shape[:2]
            # make prediction
            results = rcnn.detect([img], verbose=0)
            # get dictionary for first prediction
            result = results[0]

            # Check if any of the requested labels are found
            found_labels = [
                l for l in labels
                if class_names.index(l) in result['class_ids']
            ]
            if found_labels:
                print("Label(s) found in image: " + fileName)
                print("Annotating...")
                annotateAndSaveAnnotations(
                    result, directory_path, fileName, labels, class_names,
                    overwrite=overwrite, output_format=output_format,
                    img_width=img_width, img_height=img_height,
                    label_to_id=label_to_id)
                if display_masked:
                    display_instances(img, result['rois'], result['masks'], result['class_ids'],
                                  class_names, result['scores'])
            else:
                print("Label not found in image: " + fileName)
        except Exception as e:
            print("Error processing image {}: {}".format(fileName, e))


ROOT_DIR = os.path.abspath("./")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "./mask_rcnn_coco.h5")

# Directory to save logs, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_DATASET_LABELS = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Annotate the object')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'annotateCoco' or 'annotateCustom'")
    parser.add_argument('--image_directory', required=True,
                        metavar="/path/to/the/image/directory/",
                        help='Directory of the images that need to be annotated')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="path_to_weights.h5_file or 'coco_weights'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--label', required=True,
                        metavar="object_label_to_annotate",
                        help='Comma-separated label(s) to annotate (e.g. "person,car")')
    parser.add_argument('--labels_file',
                        metavar="/path/to/labels.txt",
                        help='File containing labels, one per line')
    parser.add_argument('--displayMaskedImages', action='store_true',
                        default=False,
                        help='Display the masked images.')
    parser.add_argument('--no-overwrite', action='store_true',
                        default=False,
                        help='Skip annotation if JSON file already exists.')
    parser.add_argument('--min_confidence', type=float, default=None,
                        metavar="0.0-1.0",
                        help='Minimum detection confidence threshold (default: model config)')
    parser.add_argument('--output_format', default='auto-annotate',
                        choices=['auto-annotate', 'coco', 'voc', 'yolo'],
                        help='Output annotation format (default: auto-annotate)')
    parser.add_argument('--device', default=None,
                        choices=['cpu', 'gpu'],
                        help='Force CPU or GPU device selection')
                        
    args = parser.parse_args()

    # Device selection
    if args.device == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
    elif args.device == 'gpu':
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            parser.error("No GPU devices available. Use --device cpu or omit --device.")

    # Parse labels (comma-separated or from file)
    labels = [l.strip() for l in args.label.split(",") if l.strip()]
    if args.labels_file:
        with open(args.labels_file, 'r') as f:
            file_labels = [line.strip() for line in f if line.strip()]
            labels.extend(file_labels)
    labels = list(dict.fromkeys(labels))  # deduplicate preserving order

    # Validate arguments
    if args.command == "annotateCoco":
        for lbl in labels:
            if lbl not in COCO_DATASET_LABELS:
                parser.error("Label '{}' does not belong to COCO labels".format(lbl))

    elif args.command == "annotateCustom":
        if not labels:
            parser.error("Argument --label is required for annotation")

    if not args.image_directory:
        parser.error("Argument --image_directory is required for annotation")
    if not args.weights:
        parser.error("Argument --weights is required for annotation")


    class InferenceCocoConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        NAME = "inferenceCoco"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80
        
    class InferenceCustomConfig(Config):
        NAME = "inferenceCustom"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 1


    if args.command == "annotateCoco":
        config = InferenceCocoConfig()
        class_names = COCO_DATASET_LABELS[:]
    else:
        config = InferenceCustomConfig()
        class_names = ['BG'] + labels

    # Override confidence threshold if specified
    if args.min_confidence is not None:
        config.DETECTION_MIN_CONFIDENCE = args.min_confidence

    config.display()

    # Create model
    model = MaskRCNN(mode="inference", config=config, model_dir="./")

    # Select weights file to load
    if args.command == "annotateCoco":
        weights_path = COCO_WEIGHTS_PATH

        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights... ", weights_path)
    model.load_weights(weights_path, by_name=True)


    # Annotate
    if args.command == "annotateCoco" or args.command == "annotateCustom":
        annotateImagesInDirectory(model, directory_path=args.image_directory,
                                  labels=labels, class_names=class_names,
                                  display_masked=args.displayMaskedImages,
                                  overwrite=not args.no_overwrite,
                                  output_format=args.output_format)
    else:
        print("'{}' is not recognized. "
              "Use 'annotateCoco' or 'annotateCustom'".format(args.command))

