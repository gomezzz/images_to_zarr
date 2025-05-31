# images_to_zarr

[![PyPI version](https://badge.fury.io/py/images-to-zarr.svg)](https://badge.fury.io/py/images-to-zarr)
[![Python](https://img.shields.io/pypi/pyversions/images-to-zarr.svg)](https://pypi.org/project/images-to-zarr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python module to efficiently bulk-convert large collections of heterogeneous images (FITS, PNG, JPEG, TIFF) into sharded Zarr v3 stores for fast analysis and cloud-native workflows.

## Features

- **Multi-format support**: FITS, PNG, JPEG, TIFF images
- **Efficient storage**: Sharded Zarr v3 format with configurable compression
- **Metadata preservation**: Combines image data with tabular metadata
- **Parallel processing**: Multi-threaded conversion for large datasets
- **Cloud-friendly**: S3-compatible storage backend
- **Easy inspection**: Built-in tools to analyze converted stores

## Installation

### From PyPI

```bash
pip install images-to-zarr
```

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

# Advanced options
images_to_zarr convert /path/to/images1 /path/to/images2 \
    --metadata metadata.csv \
    --out /output/dir \
    --recursive \
    --workers 16 \
    --fits-ext 0 \
    --chunk-shape 1,512,512 \
    --compressor zstd \
    --clevel 5 \
    --overwrite
```

Inspect a Zarr store:

```bash
images_to_zarr inspect /path/to/store.zarr
```

### Python API

```python
from images_to_zarr import convert, inspect
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

# Convert images to Zarr without metadata (filenames only)
zarr_path = convert(
    folders=["/path/to/images"],
    recursive=True,
    metadata=None,  # or simply omit this parameter
    output_dir="/output/dir"
)

# Inspect the result
inspect(zarr_path)
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

### Configuration Options

| Parameter              | Description                                     | Default       |
| ---------------------- | ----------------------------------------------- | ------------- |
| `chunk_shape`          | Zarr chunk dimensions (n_images, height, width) | (1, 256, 256) |
| `shard_bytes`          | Target shard size in bytes                      | 16 MB         |
| `compressor`           | Compression codec (zstd, lz4, gzip, etc.)       | "zstd"        |
| `clevel`               | Compression level (1-9)                         | 4             |
| `num_parallel_workers` | Number of processing threads                    | 8             |

## Output Structure

```
output_dir/
├── metadata.zarr/           # Main Zarr store
│   ├── images/             # Image data arrays
│   └── .zarray, .zgroup    # Zarr metadata
└── metadata.parquet  # Combined metadata
```

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
Compression: zstd (level 4)

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
