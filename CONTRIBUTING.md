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
   pip install -r requirements.txt
   ```

### Docker Setup (Alternative)

```bash
docker build -t auto-annotate-br .
docker run --rm -v /path/to/images:/data auto-annotate-br \
  annotateCoco --image_directory=/data --label=person --weights=/app/mask_rcnn_coco.h5
```

## Coding Conventions

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Use `os.path.join(...)` for all file path construction — never string concatenation.
- Pass dependencies (e.g., `class_names`, configuration) as explicit function parameters rather than relying on global variables.
- Use `parser.error(...)` instead of bare `assert` for CLI argument validation.
- Wrap per-image processing in `try/except` to allow batch operations to continue on individual failures.
- Use `sorted(os.listdir(...))` for deterministic file ordering.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. Make your changes in small, focused commits.
3. Ensure your changes do not break existing functionality.
4. Update documentation if your changes affect usage or configuration.
5. Open a Pull Request with a clear description of what was changed and why.

## Reporting Issues

- Use [GitHub Issues](https://github.com/VoxleOne/Auto-Annotate-BR/issues) to report bugs or request features.
- Check [knownIssues.md](knownIssues.md) for previously documented problems and workarounds.
- Include your Python version, TensorFlow version, OS, and a full stack trace when reporting bugs.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
