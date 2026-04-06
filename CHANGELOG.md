# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
