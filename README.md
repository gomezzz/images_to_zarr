# images_to_zarr

[![PyPI version](https://badge.fury.io/py/images-to-zarr.svg)](https://badge.fury.io/py/images-to-zarr)
[![Python](https://img.shields.io/pypi/pyversions/images-to-zarr.svg)](https://pypi.org/project/images-to-zarr/)

A Python module to efficiently bulk-convert large collections of heterogeneous images (FITS, PNG, JPEG, TIFF) into sharded Zarr v3 stores for fast analysis and cloud-native workflows.

## Features

- **Multi-format support**: FITS, PNG, JPEG, TIFF images
- **Consistent NCHW format**: All images stored in (batch, channels, height, width) format for ML workflows
- **Direct memory conversion**: Convert numpy arrays directly to Zarr without intermediate files
- **Efficient storage**: Sharded Zarr v3 format with configurable compression
- **Metadata preservation**: Combines image data with tabular metadata
- **Parallel processing**: Multi-threaded conversion for large datasets
- **Cloud-friendly**: S3-compatible storage backend
- **Visual inspection**: Built-in plotting tools to sample and display stored images
- **Easy inspection**: Built-in tools to analyze converted stores
- **Append functionality**: Add new images to existing Zarr stores

## Installation

### From PyPI

```bash
pip install images-to-zarr
```

After installation, the CLI command `images_to_zarr` will be available system-wide.

### From source

```bash
git clone https://github.com/gomezzz/images_to_zarr.git
cd images_to_zarr
pip install -e .
```

### Using conda

```bash
conda env create -f environment.yml
conda activate img2zarr
pip install -e .
```

## Quick Start

### Command Line Interface

Convert image folders to Zarr:

```bash
# Basic conversion with metadata
images_to_zarr convert /path/to/images --metadata metadata.csv --out /output/dir

# Basic conversion without metadata (filenames only)
images_to_zarr convert /path/to/images --out /output/dir

# Convert images to Zarr with metadata
images_to_zarr convert /path/to/images --metadata metadata.csv --out /output/dir

# Convert without metadata (filenames only)
images_to_zarr convert /path/to/images --out /output/dir

# Advanced options with resize
images_to_zarr convert /path/to/images1 /path/to/images2 \
    --metadata metadata.csv \
    --out /output/dir \
    --recursive \
    --workers 16 \
    --fits-ext 0 \
    --chunk-shape 1,512,512 \
    --compressor zstd \
    --clevel 5 \
    --resize 256,256 \
    --interpolation-order 1 \
    --overwrite

# Append new images to existing store
images_to_zarr convert /path/to/new/images \
    --metadata new_metadata.csv \
    --out /existing/store.zarr \
    --append
```

Inspect a Zarr store:

```bash
images_to_zarr inspect /path/to/store.zarr
```

### Python API

```python
from images_to_zarr import convert, inspect, display_sample_images
import numpy as np
from pathlib import Path

# Convert images to Zarr with metadata
zarr_path = convert(
    folders=["/path/to/images"],
    recursive=True,
    metadata="/path/to/metadata.csv",  # Optional
    output_dir="/output/dir",
    num_parallel_workers=8,
    chunk_shape=(1, 256, 256),
    compressor="zstd",
    clevel=4
)

# Convert images to Zarr with automatic resizing
zarr_path = convert(
    folders=["/path/to/images"],
    recursive=True,
    metadata="/path/to/metadata.csv",  # Optional
    output_dir="/output/dir",
    resize=(256, 256),  # Resize all images to 256x256
    interpolation_order=1,  # Bi-linear interpolation
    num_parallel_workers=8,
    chunk_shape=(1, 256, 256),
    compressor="zstd",
    clevel=4
)

# Convert images to Zarr without metadata (filenames only)
zarr_path = convert(
    folders=["/path/to/images"],
    recursive=True,
    metadata=None,  # or simply omit this parameter
    output_dir="/output/dir"
)

# Convert numpy arrays directly to Zarr (memory-to-zarr conversion)
# Images must be in NCHW format: (batch, channels, height, width)
images = np.random.rand(100, 3, 224, 224).astype(np.float32)  # 100 RGB images
zarr_path = convert(
    output_dir="/output/dir",
    images=images,
    compressor="lz4",
    overwrite=True
)

# Convert with custom metadata for memory conversion
metadata = [{"id": i, "source": "generated"} for i in range(100)]
zarr_path = convert(
    output_dir="/output/dir",
    images=images,
    image_metadata=metadata,
    chunk_shape=(10, 224, 224),  # Chunk 10 images together
    overwrite=True
)

# Append images to existing store
new_images = np.random.rand(50, 3, 224, 224).astype(np.float32)  # 50 more images
new_metadata = [{"id": i, "source": "appended"} for i in range(100, 150)]
zarr_path = convert(
    output_dir="/output/dir",
    images=new_images,
    image_metadata=new_metadata,
    append=True  # Append to existing store
)

# Inspect the result
inspect(zarr_path)

# Display random sample images from the store (with auto-normalization for .fits)
from images_to_zarr import display_sample_images
display_sample_images(zarr_path, num_samples=6, figsize=(15, 10))

# Save sample images to file
display_sample_images(zarr_path, num_samples=4, save_path="samples.png")

# Append more images from memory
new_images = np.random.rand(25, 3, 224, 224).astype(np.float32)
zarr_path = convert(
    output_dir="/output/dir",
    images=new_images,
    append=True  # Append to existing store
)
```

## Usage

### Metadata CSV Format

The metadata CSV file is **optional**. If provided, it must contain at least a `filename` column. Additional columns are preserved:

```csv
filename,source_id,ra,dec,magnitude
image001.fits,12345,123.456,45.678,18.5
image002.png,12346,124.567,46.789,19.2
image003.jpg,12347,125.678,47.890,17.8
```

If no metadata file is provided, metadata will be automatically created from the filenames:

```bash
# Convert without metadata - will use filenames only
images_to_zarr convert /path/to/images --out /output/dir

# Convert with metadata
images_to_zarr convert /path/to/images --metadata metadata.csv --out /output/dir
```

### Supported Image Formats

- **FITS** (`.fits`, `.fit`): Astronomical images with flexible HDU support
- **PNG** (`.png`): Lossless compressed images
- **JPEG** (`.jpg`, `.jpeg`): Compressed photographic images  
- **TIFF** (`.tif`, `.tiff`): Uncompressed or losslessly compressed images

### FITS Extension Handling

```python
# Use primary HDU (default)
convert(..., fits_extension=None)

# Use specific extension by number
convert(..., fits_extension=1)

# Use extension by name
convert(..., fits_extension="SCI")

# Combine multiple extensions
convert(..., fits_extension=[0, 1, "ERR"])
```

### Image Resizing

When dealing with images of different sizes, you can use the resize functionality:

```python
# Resize all images to 512x512 using bi-linear interpolation
convert(
    folders=["/path/to/images"],
    output_dir="/output/dir",
    resize=(512, 512),
    interpolation_order=1  # 0=nearest, 1=linear, 2=quadratic, etc.
)

# If resize is not specified, all images must have the same dimensions
# or an error will be raised
```

**Interpolation orders:**
- 0: Nearest-neighbor (fastest, lowest quality)
- 1: Bi-linear (default, good balance)
- 2: Bi-quadratic
- 3: Bi-cubic (slower, higher quality)
- 4: Bi-quartic
- 5: Bi-quintic (slowest, highest quality)

### Configuration Options

| Parameter              | Description                                     | Default       |
| ---------------------- | ----------------------------------------------- | ------------- |
| `chunk_shape`          | Zarr chunk dimensions (n_images, height, width) | (1, 256, 256) |
| `compressor`           | Compression codec (zstd, lz4, gzip, etc.)       | "lz4"         |
| `clevel`               | Compression level (1-9)                         | 1             |
| `num_parallel_workers` | Number of processing threads                    | 8             |
| `recursive`            | Scan subdirectories recursively                 | False         |
| `fits_extension`       | FITS HDU(s) to read (int, str, or sequence)     | None (uses 0) |
| `resize`               | Resize images to (height, width)                | None          |
| `interpolation_order`  | Resize interpolation order (0-5)                | 1 (bi-linear) |
| `overwrite`            | Overwrite existing store if present             | False         |
| `append`               | Append to existing store                         | False         |

## Append Functionality

You can add new images to existing Zarr stores using the `append=True` parameter. This is useful for:

- **Incremental data processing**: Add new images as they become available
- **Distributed processing**: Combine results from multiple processing nodes
- **Large dataset management**: Build up large datasets incrementally

### Append Requirements

- **Compatible dimensions**: New images must have the same shape as existing images (except batch dimension)
- **Compatible data types**: New images are automatically converted to match existing store dtype
- **Mutually exclusive with overwrite**: Cannot use `append=True` and `overwrite=True` together

### Append Examples

```python
from images_to_zarr import convert

# Create initial store
initial_images = np.random.rand(100, 3, 256, 256).astype(np.float32)
zarr_path = convert(
    output_dir="./dataset.zarr",
    images=initial_images,
    overwrite=True
)

# Append more images
additional_images = np.random.rand(50, 3, 256, 256).astype(np.float32)
convert(
    output_dir="./dataset.zarr",
    images=additional_images,
    append=True  # Append to existing store
)

# Result: dataset.zarr now contains 150 images (100 + 50)
```

### Append with File-based Conversion

```bash
# Create initial store
images_to_zarr convert /initial/images --out /dataset.zarr

# Append more images later
images_to_zarr convert /new/images --out /dataset.zarr --append
```

### Append History

Each append operation is tracked in the Zarr store attributes:

```python
import zarr

store = zarr.storage.LocalStore("./dataset.zarr")
root = zarr.open_group(store=store, mode="r")
print(root.attrs["append_history"])
# [{"appended_count": 50, "start_index": 100, "end_index": 150}]
```

## Output Structure

```
output_dir/
├── images.zarr/              # Main Zarr store (if output_dir doesn't end with .zarr)
│   ├── images/              # Image data arrays
│   └── .zarray, .zgroup     # Zarr metadata
└── images_metadata.parquet  # Combined metadata
```

**Note**: If you specify an output directory ending with `.zarr` (e.g., `/path/to/my_dataset.zarr`), 
that path will be used directly as the Zarr store, creating a cleaner output structure.

### Zarr Store Contents

- **`images`**: Main array containing all image data
- **Attributes**: Store metadata, compression info, creation parameters
- **Chunks**: Sharded for efficient cloud access

### Metadata Parquet

Combined metadata includes:
- Original CSV columns
- Image-specific metadata (dimensions, dtype, file size)
- Processing statistics (min/max/mean values)

## Performance Tips

1. **Chunk size**: Match your typical access patterns
   - Single image access: `(1, H, W)`
   - Batch processing: `(B, H, W)` where B > 1

2. **Compression**: Balance speed vs. size
   - Fast: `lz4` with low compression level
   - Compact: `zstd` with high compression level

3. **Parallelism**: Scale with your I/O capacity
   - Local SSD: 8-16 workers
   - Network storage: 4-8 workers
   - S3: 16-32 workers

4. **Memory**: Monitor for large images
   - Consider smaller chunk sizes for very large images
   - Reduce batch size if memory usage is high

## Inspection Output Example

```
================================================================================
SUMMARY STATISTICS  
================================================================================
Total images across all files: 104,857,600
Total storage size: 126,743.31 MB
Image dimensions: (3, 256, 256)
Data type: uint8
Compression: lz4 (level 1)

Format distribution:
  FITS: 60,000,000 (57.2%)
  PNG: 30,000,000 (28.6%) 
  JPEG: 10,000,000 (9.5%)
  TIFF: 4,857,600 (4.6%)

Original data type distribution:
  uint8: 78.0%
  int16: 12.0%
  float32: 10.0%
================================================================================
```

## Image Display and Visualization

The `display_sample_images` function provides intelligent visualization with automatic normalization:

```python
from images_to_zarr import display_sample_images

# Display with automatic normalization (handles .fits files with arbitrary ranges)
display_sample_images("/path/to/store.zarr", num_samples=6)
```


## Error Handling

The library provides robust error handling:

- **Missing files**: Warnings logged, processing continues
- **Corrupted images**: Replaced with zero arrays, errors recorded in metadata  
- **Incompatible formats**: Clear error messages with suggested fixes
- **Storage issues**: Detailed error reporting for disk/network problems

## Logging Configuration

```python
from images_to_zarr import configure_logging

# Enable detailed logging
configure_logging(enable=True, level="DEBUG")

# Disable for production
configure_logging(enable=False)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Setup

```bash
git clone https://github.com/username/images_to_zarr.git
cd images_to_zarr
conda env create -f environment.yml
conda activate img2zarr
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Check linting
flake8
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [Zarr](https://zarr.readthedocs.io/) for array storage
- Uses [Astropy](https://www.astropy.org/) for FITS support
- Inspired by the needs of astronomical data processing pipelines

### Channel Order and Format Consistency

All images are automatically converted to **NCHW format** (batch, channels, height, width) for consistency across different input formats:

- **2D grayscale**: `(H, W)` → `(1, 1, H, W)`
- **3D RGB (HWC)**: `(H, W, C)` → `(1, C, H, W)` 
- **3D CHW**: `(C, H, W)` → `(1, C, H, W)`
- **4D batched**: Already in NCHW format

The library intelligently detects the input format:
- Images with ≤4 channels in the last dimension are treated as HWC (Height-Width-Channels)
- Images with >4 channels in the last dimension are treated as CHW (Channels-Height-Width)
- FITS files and other scientific formats are handled appropriately

This ensures consistent tensor shapes for machine learning workflows while preserving the original data.

### Direct Memory Conversion

Convert numpy arrays directly to Zarr without saving intermediate files:

```python
import numpy as np
from images_to_zarr import convert

# Your image data (must be 4D NCHW format)
images = np.random.rand(1000, 3, 256, 256).astype(np.float32)

# Convert directly to zarr
zarr_path = convert(
    output_dir="./data",
    images=images,
    compressor="lz4",
    chunk_shape=(100, 256, 256),  # Chunk 100 images together
    overwrite=True
)
```
