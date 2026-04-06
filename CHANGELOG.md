# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.0.0] - 2026-04-06

### Added
- **YOLOv8 backend** as default detection backend (`--backend yolov8`), powered by the `ultralytics` package and PyTorch.
- **Backend abstraction layer** (`backends/` module) with `DetectionBackend` ABC and `DetectionResult` dataclass, allowing pluggable detection backends.
- `backends/yolov8.py`: YOLOv8 inference and segmentation via Ultralytics, supporting model sizes nano/small/medium/large/xlarge.
- `backends/maskrcnn.py`: Legacy Mask R-CNN backend wrapping the vendored `mrcnn/` module.
- `--backend` CLI argument for both `annotate.py` and `customTrain.py` to select detection backend.
- `--model_size` CLI argument for YOLOv8 model size selection.
- `--epochs` CLI argument for `customTrain.py` to control training duration.
- `--label` CLI argument for `customTrain.py` to specify the custom class name.
- `convert` command in `customTrain.py` to convert VIA JSON annotations to YOLO `.txt` format.
- `requirements-maskrcnn.txt` for optional legacy Mask R-CNN dependencies.
- `Dockerfile.maskrcnn` for building images with both backends.
- `PyYAML` dependency for YOLO dataset config generation.

### Changed
- **Breaking**: Default backend changed from Mask R-CNN to YOLOv8. Use `--backend maskrcnn` to keep previous behavior.
- **Breaking**: `COCO_DATASET_LABELS` no longer includes `'BG'` prefix. The legacy list with BG is available as `COCO_DATASET_LABELS_WITH_BG`.
- **Breaking**: `annotateImagesInDirectory()` now takes `(backend, model, ...)` instead of `(rcnn, ...)`. Functions accept `DetectionResult` lists instead of raw Mask R-CNN result dicts.
- `Dockerfile` base image changed from `tensorflow/tensorflow:2.10.0-gpu` to `python:3.10-slim`.
- `requirements.txt` now installs `ultralytics` and `torch` by default. TensorFlow dependencies are optional (see `requirements-maskrcnn.txt`).
- Image loading uses PIL directly instead of `tensorflow.keras.preprocessing.image`.
- `customTrain.py` uses lazy imports for Mask R-CNN modules to avoid requiring TensorFlow at import time.

### Deprecated
- The Mask R-CNN backend (`--backend maskrcnn`) is deprecated and will be removed in a future version. A deprecation warning is emitted when it is used.

### Removed
- `--displayMaskedImages` CLI flag (was Mask R-CNN specific; may be re-added in a future release for all backends).
- Top-level TensorFlow imports from `annotate.py` (moved to backend module).

## [2.0.0] - 2026-04-06

### Changed
- **Breaking**: Migrated from standalone Keras + TensorFlow 1.x to TensorFlow 2.x with `tensorflow.keras`. All `keras.*` imports replaced with `tensorflow.keras.*`.
- **Breaking**: Replaced `tf.logging.set_verbosity` with `tf.get_logger().setLevel('ERROR')`.
- **Breaking**: Replaced `distutils.version.LooseVersion` with `packaging.version.Version`.
- **Breaking**: Replaced deprecated `np.bool` with Python built-in `bool`.
- Updated `requirements.txt`: pinned `tensorflow>=2.10,<2.16`, `numpy>=1.23,<2.0`, `h5py>=3.1`, `scikit-image>=0.19`, `Shapely>=2.0`. Removed `mrcnn==0.2` and `keras==2.2.4`.
- `--displayMaskedImages` now uses `action='store_true'` (pass the flag without a value).
- Functions `annotateResult`, `create_sub_mask_annotation`, `annotateAndSaveAnnotations`, and `annotateImagesInDirectory` now accept `class_names` as an explicit parameter instead of relying on global state.
- File paths constructed with `os.path.join(...)` instead of string concatenation.
- Directory listing uses `sorted(os.listdir(...))` for deterministic ordering.
- CLI validation uses `parser.error(...)` instead of bare `assert`.
- Renamed `bugsrelatados.md` → `knownIssues.md`.

### Added
- `Dockerfile` based on `tensorflow/tensorflow:2.10.0-gpu`.
- `--no-overwrite` CLI flag to skip annotation if JSON file already exists.
- `try/except` around per-image inference so batch processing continues on failures.
- `CONTRIBUTING.md` with setup instructions, coding conventions, and PR process.
- `CHANGELOG.md` (this file).
- Multi-label annotation support via comma-separated `--label` values.
- `--min_confidence` CLI argument to control detection confidence threshold.
- `--device cpu|gpu` CLI argument for explicit device selection.
- `tqdm` progress bar for batch image processing.
- Multiple output formats: `--output_format` supports `auto-annotate` (default), `coco`, `voc`, and `yolo`.

### Fixed
- `ROOT_DIR` in `customTrain.py` changed from `../../` to `./`.
- `image_reference()` copy-paste bug: source check changed from `"Bird House"` to `"customLabel"`; added missing `return` in `else` branch.
- Removed dead `splash` command from `customTrain.py`.

### Removed
- Unused imports: `find_contours` (line 2) and `matplotlib.pyplot` in `annotate.py`.
- `mrcnn==0.2` from `requirements.txt` (the vendored `mrcnn/` directory is used instead).
- Dead `--image` and `--video` arguments from `customTrain.py`.

## [1.0.0] - 2021-01-01

### Added
- Initial fork of [mdhmz1/Auto-Annotate](https://github.com/mdhmz1/Auto-Annotate) with Brazilian Portuguese translation.
- COCO and custom label annotation modes.
- JSON output format for segmentation annotations.
