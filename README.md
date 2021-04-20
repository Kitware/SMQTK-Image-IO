# SMQTK - Image-IO

## Intent
This package is intended to provide the interfaces and tools for working with image input.

Specifically, there are two readers used to interact with images. One uses GDAL and the other uses PIL

## Documentation

You can build the sphinx documentation locally for the most
up-to-date reference (see also: [Building the Documentation](
https://smqtk.readthedocs.io/en/latest/installation.html#building-the-documentation)):
```bash
# Install dependencies
poetry install
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```
