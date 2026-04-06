# Contributing to Auto-Annotate-BR

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- **Python 3.9 or 3.10** (recommended)
- **pip** or **conda** for package management

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/VoxleOne/Auto-Annotate-BR.git
   cd Auto-Annotate-BR
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   # YOLOv8 backend only (default)
   pip install -r requirements.txt

   # Both backends (includes TensorFlow for legacy Mask R-CNN)
   pip install -r requirements.txt -r requirements-maskrcnn.txt
   ```

### Docker Setup (Alternative)

```bash
# YOLOv8 only
docker build -t auto-annotate-br .

# Both backends
docker build -f Dockerfile.maskrcnn -t auto-annotate-br-full .

docker run --rm -v /path/to/images:/data auto-annotate-br \
  annotateCoco --image_directory=/data --label=person --weights=coco
```

## Architecture: Backend Abstraction

The project uses a **backend abstraction pattern** to support multiple
detection frameworks.  All backends live in the `backends/` module.

### Key components

- **`backends/__init__.py`** — `DetectionBackend` ABC and `DetectionResult`
  dataclass.  The factory function `get_backend(name)` returns the
  correct backend instance.
- **`backends/yolov8.py`** — YOLOv8 backend (Ultralytics).  Default.
- **`backends/maskrcnn.py`** — Legacy Mask R-CNN backend (TensorFlow).

### Adding a new backend

1. Create `backends/my_backend.py` with a class that extends
   `DetectionBackend`.
2. Implement `load_model()`, `detect()`, and `get_class_names()`.
3. Register it in `get_backend()` inside `backends/__init__.py`.
4. Add the new choice to the `--backend` argument in `annotate.py`
   and `customTrain.py`.

### `DetectionResult` contract

Every backend returns a list of `DetectionResult` dataclass instances:

```python
@dataclass
class DetectionResult:
    class_id: int            # 0-based class ID (no background)
    label: str               # Human-readable class name
    bbox: tuple              # (x, y, width, height) absolute pixels
    score: float             # Confidence score 0–1
    mask: np.ndarray | None  # Binary mask (H, W) or None
```

## Coding Conventions

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Use `os.path.join(...)` for all file path construction — never string concatenation.
- Pass dependencies (e.g., `class_names`, configuration) as explicit function parameters rather than relying on global variables.
- Use `parser.error(...)` instead of bare `assert` for CLI argument validation.
- Wrap per-image processing in `try/except` to allow batch operations to continue on individual failures.
- Use `sorted(os.listdir(...))` for deterministic file ordering.
- Lazy-import heavy frameworks (TensorFlow, PyTorch) inside functions or backend modules, not at the top of `annotate.py` or `customTrain.py`.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. Make your changes in small, focused commits.
3. Ensure your changes do not break existing functionality.
4. Update documentation if your changes affect usage or configuration.
5. Open a Pull Request with a clear description of what was changed and why.

## Reporting Issues

- Use [GitHub Issues](https://github.com/VoxleOne/Auto-Annotate-BR/issues) to report bugs or request features.
- Check [knownIssues.md](knownIssues.md) for previously documented problems and workarounds.
- Include your Python version, backend (`yolov8` or `maskrcnn`), OS, and a full stack trace when reporting bugs.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
