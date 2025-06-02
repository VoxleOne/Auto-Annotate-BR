import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf # TF2 Change: Ensure TensorFlow is imported

# Root directory of the project
# TF2 Change: Consider making ROOT_DIR more robust if script is moved
# For example, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Assuming script is in a subdir of the main project (e.g. /samples/custom/)
# If ROOT_DIR is meant to be the parent of 'mrcnn' dir:
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_SCRIPT_DIR) # If mrcnn is in the same dir as customTrain.py's parent
# Or, if customTrain.py is at the same level as the mrcnn folder:
# ROOT_DIR = os.path.abspath(".")
# The original ROOT_DIR = os.path.abspath("../../") implies a structure like /some_project/samples/custom/customTrain.py
# and mrcnn is in /some_project/mrcnn.
# For now, let's assume the existing ROOT_DIR logic is correct for the project structure.
# If mrcnn is not found, this sys.path.append will be critical.
# ROOT_DIR = os.path.abspath("../../") # Original, keep if structure matches
# Let's try a more common setup where mrcnn is a subdir or at same level
if os.path.exists(os.path.join(os.path.abspath("."), "mrcnn")):
    ROOT_DIR = os.path.abspath(".")
elif os.path.exists(os.path.join(os.path.abspath(".."), "mrcnn")):
    ROOT_DIR = os.path.abspath("..")
elif os.path.exists(os.path.join(os.path.abspath("../.."), "mrcnn")): # Original assumption
    ROOT_DIR = os.path.abspath("../..")
else:
    # Fallback or error if mrcnn path is not easily found
    print("Warning: mrcnn library path not automatically detected. Assuming ROOT_DIR is two levels up.")
    ROOT_DIR = os.path.abspath("../..")


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
# TF2 Change: modellib and utils are already refactored
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") # TF2 Change: Ensure this path is correct

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "CustomLabel"

    # Adjust batch size based on GPU availability and strategy
    # IMAGES_PER_GPU will be used by the data generator.
    # The effective batch size for training will be IMAGES_PER_GPU * number of GPUs.
    IMAGES_PER_GPU = 1 # Can be increased if GPU memory allows

    # TF2 Change: GPU_COUNT is still used to determine strategy.
    # It's set by Config based on available GPUs by default.
    # We can override it here if needed, or let the Config class detect it.
    # For explicit control with MirroredStrategy:
    # GPU_COUNT = 2 # For example, if you want to use 2 GPUs

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + customLabel

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # Optional: Set specific GPU count if auto-detection is not desired
    # GPU_COUNT = 1 # or 2, etc. Default in Config class tries to detect.


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("CustomLabel", 1, "CustomLabel")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir_subset = os.path.join(dataset_dir, subset) # Use a different var name

        # Load annotations
        annotations_path = os.path.join(dataset_dir_subset, "via_region_data.json")
        try:
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
        except FileNotFoundError:
            print(f"Error: Annotation file not found at {annotations_path}")
            print("Please ensure your dataset is structured correctly with 'train' and 'val' subdirectories,")
            print("each containing a 'via_region_data.json' file and the corresponding images.")
            sys.exit(1)
            
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            image_path = os.path.join(dataset_dir_subset, a['filename'])
            try:
                image = skimage.io.imread(image_path)
            except FileNotFoundError:
                print(f"Warning: Image file {a['filename']} not found at {image_path}. Skipping this image.")
                continue
                
            height, width = image.shape[:2]

            self.add_image(
                "CustomLabel", # Source name matches add_class
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "CustomLabel":
            return super(CustomDataset, self).load_mask(image_id) # Python 3 super()

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # Ensure p['all_points_y'] and p['all_points_x'] exist and are not empty
            if not (p.get('all_points_y') and p.get('all_points_x')):
                print(f"Warning: Skipping polygon {i} for image {info['id']} due to missing coordinate data.")
                continue
            try:
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                # Ensure rr and cc are within mask bounds
                rr = np.clip(rr, 0, info["height"] - 1)
                cc = np.clip(cc, 0, info["width"] - 1)
                mask[rr, cc, i] = 1
            except Exception as e:
                print(f"Error drawing polygon for image {info['id']}, polygon {i}: {e}")
                print(f"Polygon data: Y={p.get('all_points_y')}, X={p.get('all_points_x')}")
                # Optionally, skip this polygon or image
                continue


        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32) # TF2 Change: np.bool -> bool

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        # TF2 Change: Original had "Bird House", changed to "CustomLabel" for consistency
        if info["source"] == "CustomLabel": 
            return info["path"]
        else:
            return super(CustomDataset, self).image_reference(image_id) # Python 3 super()


def train(model_instance, config_instance): # Pass config instance
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    print("Training network heads")
    model_instance.train(dataset_train, dataset_val,
                learning_rate=config_instance.LEARNING_RATE, # Use config_instance
                epochs=args.epochs if hasattr(args, 'epochs') else 30, # Use args.epochs if provided
                layers='heads',
                # Augmentation can be added here if desired
                # augmentation=imgaug.augmenters.Sometimes(0.5, [
                #     imgaug.augmenters.Fliplr(0.5),
                #     imgaug.augmenters.GaussianBlur(sigma=(0.0, 1.0))
                # ])
                )
    
    # Example for training all layers after heads
    # print("Fine tune Resnet stage 4 and up")
    # model_instance.train(dataset_train, dataset_val,
    #             learning_rate=config_instance.LEARNING_RATE / 10,
    #             epochs=args.epochs_all if hasattr(args, 'epochs_all') else 60, # Example: another arg for total epochs
    #             layers='4+',
    #             augmentation=None) # Add augmentation if needed


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'") # 'splash' part is not implemented in this script
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the Custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco', 'last', 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    # TF2 Change: Add epochs argument for more flexibility
    parser.add_argument('--epochs', required=False, type=int, default=30,
                        help='Number of epochs to train for heads (default=30)')
    # parser.add_argument('--epochs_all', required=False, type=int, default=60,
    #                     help='Number of epochs to train for all layers (default=60)')

    # Arguments for splash (not implemented here but kept for parser structure)
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash": # Splash functionality is not in this script
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
        print("Splash command is not fully implemented in this version of customTrain.py")
        sys.exit()


    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = CustomConfig() # Initialize config first
    # TF2 Change: Determine and set GPU strategy
    if config.GPU_COUNT > 1:
        print(f"Attempting to use {config.GPU_COUNT} GPUs.")
        try:
            # Get available physical GPUs
            physical_gpus = tf.config.list_physical_devices('GPU')
            if len(physical_gpus) < config.GPU_COUNT:
                print(f"Warning: Config.GPU_COUNT is {config.GPU_COUNT}, but only {len(physical_gpus)} physical GPUs are available.")
                print(f"Using {len(physical_gpus)} available GPUs instead.")
                # Optionally adjust config.GPU_COUNT here if you want the strategy to use fewer GPUs
                # For MirroredStrategy, it will use all available by default if no specific devices are passed.
                # Or, you can explicitly tell it which GPUs to use.
                # For simplicity, let's let MirroredStrategy use all available if config.GPU_COUNT > 1
            
            if physical_gpus: # Only create strategy if GPUs are actually available
                 strategy = tf.distribute.MirroredStrategy()
                 print(f"Running with MirroredStrategy on {strategy.num_replicas_in_sync} replicas (GPUs).")
                 # Adjust IMAGES_PER_GPU if needed, or ensure BATCH_SIZE in config is global batch size
                 # config.BATCH_SIZE = config.IMAGES_PER_GPU * strategy.num_replicas_in_sync
            else:
                print("Warning: No GPUs detected by TensorFlow. Running on CPU.")
                strategy = tf.distribute.get_strategy() # Default strategy (usually single CPU)
                config.GPU_COUNT = 0 # Reflect that no GPUs are used
                config.IMAGES_PER_GPU = config.BATCH_SIZE # Ensure BATCH_SIZE is IMAGES_PER_GPU for CPU

        except RuntimeError as e:
            print(f"Error initializing MirroredStrategy: {e}. Falling back to default strategy.")
            strategy = tf.distribute.get_strategy()
            config.GPU_COUNT = 0 # Or 1 if one GPU is available but strategy failed
            config.IMAGES_PER_GPU = config.BATCH_SIZE

    else: # Single GPU or CPU
        print("Using default strategy (single GPU or CPU).")
        strategy = tf.distribute.get_strategy()
        # Ensure BATCH_SIZE in config matches IMAGES_PER_GPU if on CPU/single GPU
        if config.GPU_COUNT == 0 or len(tf.config.list_physical_devices('GPU')) == 0 :
            config.IMAGES_PER_GPU = config.BATCH_SIZE
        # If GPU_COUNT is 1, BATCH_SIZE = IMAGES_PER_GPU * 1 is fine.

    config.display()


    # Create model and load weights under strategy scope
    with strategy.scope():
        if args.command == "train":
            model = modellib.MaskRCNN(mode="training", config=config,
                                      model_dir=args.logs)
        # else: # For inference mode, if you were to add it
        #     model = modellib.MaskRCNN(mode="inference", config=config,
        #                               model_dir=args.logs)

        # Select weights file to load
        if args.weights.lower() == "coco":
            weights_path = COCO_WEIGHTS_PATH
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        elif args.weights.lower() == "last":
            weights_path = model.find_last()
        elif args.weights.lower() == "imagenet":
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = args.weights

        # Load weights
        print("Loading weights from: ", weights_path)
        if args.weights.lower() == "coco":
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
        
        # Note: model.compile() is called within model.train().
        # Since 'model' (self.keras_model) was created in strategy.scope(),
        # the compilation will also be under the strategy.

    # Train or evaluate
    if args.command == "train":
        train(model, config) # Pass model and config
    else:
        print(f"'{args.command}' is not a recognized command in this script version.")
        print("Use 'train'")
