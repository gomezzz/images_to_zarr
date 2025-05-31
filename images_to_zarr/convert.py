from __future__ import annotations
from pathlib import Path
from typing import Sequence
import pandas as pd
import numpy as np
import zarr
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from loguru import logger
import imageio
from astropy.io import fits
from PIL import Image

from images_to_zarr import I2Z_SUPPORTED_EXTS


def _find_image_files(
    folders: Sequence[Path] | Sequence[str], recursive: bool = False
) -> list[Path]:
    """Find all supported image files in the given folders."""
    image_files = []

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"Folder does not exist: {folder_path}")
            continue

        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in folder_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in I2Z_SUPPORTED_EXTS:
                image_files.append(file_path)

    logger.info(f"Found {len(image_files)} image files")
    return sorted(image_files)


def _read_image_data(
    image_path: Path, fits_extension: int | str | Sequence[int | str] | None = None
) -> tuple[np.ndarray, dict]:
    """Read image data from various formats with minimal overhead."""
    file_ext = image_path.suffix.lower()

    # Essential metadata for tests and functionality
    metadata = {
        "original_filename": image_path.name,
        "original_extension": file_ext,
    }

    try:
        if file_ext in {".fits", ".fit"}:
            # Handle FITS files - simplified for speed
            if fits_extension is None:
                fits_extension = 0  # Default to first extension

            with fits.open(image_path) as hdul:
                if isinstance(fits_extension, (list, tuple)):
                    # Concatenate multiple extensions (keep existing logic)
                    arrays = []
                    for ext in fits_extension:
                        if hdul[ext].data is not None:
                            arrays.append(hdul[ext].data)
                    if not arrays:
                        raise ValueError(f"No valid data found in FITS extensions {fits_extension}")
                    data = np.concatenate(arrays, axis=0 if len(arrays[0].shape) == 2 else -1)
                    metadata["fits_extensions"] = list(fits_extension)
                else:
                    data = hdul[fits_extension].data
                    if data is None:
                        raise ValueError(f"No data found in FITS extension {fits_extension}")
                    metadata["fits_extension"] = fits_extension

        else:
            # Handle other image formats efficiently
            if file_ext in {".png", ".jpg", ".jpeg"}:
                # Use PIL for better format support
                with Image.open(image_path) as img:
                    data = np.array(img)
                    metadata["mode"] = img.mode
            else:
                # Use imageio for TIFF and other formats
                data = imageio.imread(image_path)

        # Minimal dimension handling
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 3:
            logger.warning(f"Image {image_path} has {data.ndim} dimensions, flattening extra dims")
            data = data.reshape(data.shape[0], -1)

        # Essential metadata for functionality
        metadata.update(
            {
                "dtype": str(data.dtype),
                "shape": data.shape,
            }
        )

        return data, metadata

    except Exception as e:
        logger.error(f"Failed to read {image_path}: {e}")
        raise


def _process_single_image(
    image_path: Path,
    target_shape: tuple,
    target_dtype: np.dtype,
    fits_extension: int | str | Sequence[int | str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Process a single image efficiently."""
    try:
        data, metadata = _read_image_data(image_path, fits_extension)

        # Handle different image dimensions by padding/cropping to match zarr shape
        if len(data.shape) == 2:
            # Add channel dimension if needed
            if len(target_shape) == 3:
                data = data[np.newaxis, :, :]

        # Efficient resize/crop without creating full zeros array
        final_data = np.zeros(target_shape, dtype=target_dtype)

        # Copy data with appropriate slicing
        slices = tuple(slice(0, min(s, t)) for s, t in zip(data.shape, target_shape))
        final_data[slices] = data[slices]

        return final_data, metadata

    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")
        # Create dummy data for failed images
        dummy_data = np.zeros(target_shape, dtype=target_dtype)
        return dummy_data, {
            "original_filename": image_path.name,
            "error": str(e),
            "dtype": str(dummy_data.dtype),
            "shape": dummy_data.shape,
        }


def _process_image_batch_worker(
    image_paths_batch: list[Path],
    array_shape: tuple,
    array_dtype: np.dtype,
    fits_extension: int | str | Sequence[int | str] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Process a batch of images in a separate process - worker function."""
    target_shape = array_shape[1:]  # Skip the first dimension (image index)
    batch_size = len(image_paths_batch)
    batch_metadata = []

    # Pre-allocate batch data for efficient writing
    batch_data = np.zeros((batch_size,) + target_shape, dtype=array_dtype)

    # Process images sequentially within batch (I/O bound)
    for i, image_path in enumerate(image_paths_batch):
        data, metadata = _process_single_image(
            image_path, target_shape, array_dtype, fits_extension
        )
        batch_data[i] = data
        batch_metadata.append(metadata)

    return batch_data, batch_metadata


def _process_image_batch(
    image_paths: list[Path],
    zarr_array: zarr.Array,
    start_idx: int,
    fits_extension: int | str | Sequence[int | str] | None = None,
) -> list[dict]:
    """Process a batch of images and write to zarr array efficiently."""
    # Process in same thread to avoid pickle issues with zarr arrays
    target_shape = zarr_array.shape[1:]  # Skip the first dimension (image index)
    target_dtype = zarr_array.dtype
    batch_metadata = []

    # Pre-allocate batch data for efficient writing
    batch_size = len(image_paths)
    batch_data = np.zeros((batch_size,) + target_shape, dtype=target_dtype)

    # Process images sequentially within batch (I/O bound)
    for i, image_path in enumerate(image_paths):
        data, metadata = _process_single_image(
            image_path, target_shape, target_dtype, fits_extension
        )
        batch_data[i] = data
        batch_metadata.append(metadata)

    # Single batch write to Zarr (much more efficient)
    zarr_array[start_idx : start_idx + batch_size] = batch_data

    return batch_metadata


def convert(
    folders: Sequence[Path] | Sequence[str],
    output_dir: Path | str,
    metadata: Path | str | None = None,
    recursive: bool = False,
    num_parallel_workers: int = 8,
    fits_extension: int | str | Sequence[int | str] | None = None,
    *,
    chunk_shape: tuple[int, int, int] = (1, 256, 256),
    compressor: str = "lz4",  # Changed default to fastest compressor
    clevel: int = 1,  # Changed default to fastest compression level
    overwrite: bool = False,
) -> Path:
    """
    Re-package a heterogeneous image collection (FITS/PNG/JPEG/TIFF) plus
    tabular metadata into a *single* **sharded Zarr v3** store.

    Parameters
    ----------
    folders
        One or more directories containing images.
    recursive
        If *True*, scan sub-directories too.
    metadata
        Optional CSV file with at least a ``filename`` column; additional fields
        (e.g. ``source_id``, ``ra``, ``dec`` …) are copied verbatim into
        a Parquet side-car and attached as Zarr attributes for easy joins.
        If not provided, metadata will be created from just the filenames.
    output_dir
        Destination path; a directory called ``<name>.zarr`` is created
        inside it.  Existing stores are refused unless *overwrite* is set.
    num_parallel_workers
        Threads or processes used to ingest images and write chunks.
    fits_extension
        Which FITS HDU(s) to read:

        * ``None``  →  use extension 0
        * *int* or *str*  →  single HDU
        * *Sequence*  →  concatenate multiple HDUs along the channel axis
    chunk_shape
        Chunk layout **(n_images, height, width)** ; the first dimension
        **must be 1** so each image maps to exactly one chunk.
    shard_bytes
        Target size (bytes) of each shard container file.
    compressor
        Any *numcodecs* codec name (``"zstd"``, ``"lz4"``, …).
    clevel
        Compression level handed to *numcodecs*.
    overwrite
        Destroy an existing store at *output_dir* if present.

    Returns
    -------
    Path
        Path to the root of the new ``*.zarr`` store.

    Notes
    -----
    * The function is purely I/O bound; if the host has a fast network
      file-system prefer a *ThreadPoolExecutor*.
    * A sibling file ``metadata.parquet`` is always written – fast joins,
      Arrow-native.
    * Sharding keeps the inode count roughly equal to "1 000 HDF5 files"
      for 100 M images but remains S3-friendly.
    """
    logger.info("Starting image to Zarr conversion")

    # Convert inputs to Path objects
    output_dir = Path(output_dir)

    # Find all image files
    image_files = _find_image_files(folders, recursive)
    if not image_files:
        raise ValueError("No image files found in specified folders")

    # Load or create metadata
    if metadata is not None:
        metadata_path = Path(metadata)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata_df = pd.read_csv(metadata_path)
        if "filename" not in metadata_df.columns:
            raise ValueError("Metadata CSV must contain a 'filename' column")

        store_name = f"{metadata_path.stem}.zarr"
    else:
        # Create metadata from filenames only
        metadata_df = pd.DataFrame({"filename": [img_path.name for img_path in image_files]})
        store_name = "images.zarr"
    zarr_path = output_dir / store_name

    if zarr_path.exists():
        if overwrite:
            import shutil

            shutil.rmtree(zarr_path)
            logger.info(f"Removed existing store: {zarr_path}")
        else:
            raise FileExistsError(f"Store already exists: {zarr_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine image dimensions by sampling a few files efficiently
    logger.info("Analyzing image dimensions...")
    sample_size = min(3, len(image_files))  # Reduced sample size for speed
    max_height, max_width = 224, 224  # Assume common size, adjust if needed
    max_channels = 3
    sample_dtype = np.uint8

    for img_path in image_files[:sample_size]:
        try:
            # Quick dimension check without full processing
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                with Image.open(img_path) as img:
                    w, h = img.size
                    c = 3 if img.mode == "RGB" else 1
                    if img.mode in ["I", "I;16"]:
                        sample_dtype = np.uint16
                    elif img.mode == "F":
                        sample_dtype = np.float32
            else:
                data, _ = _read_image_data(img_path, fits_extension)
                if len(data.shape) == 2:
                    h, w = data.shape
                    c = 1
                elif len(data.shape) == 3:
                    c, h, w = data.shape
                else:
                    continue

                # Use the most general dtype
                if np.issubdtype(data.dtype, np.floating):
                    sample_dtype = np.float32
                elif data.dtype == np.uint16:
                    sample_dtype = np.uint16

            max_height = max(max_height, h)
            max_width = max(max_width, w)
            max_channels = max(max_channels, c)

        except Exception as e:
            logger.warning(f"Could not analyze {img_path}: {e}")
            continue

    # Adjust chunk shape to match data dimensions - optimize for parallel access
    if max_channels > 1:
        array_shape = (len(image_files), max_channels, max_height, max_width)
        # Chunk multiple images together for better compression and I/O
        chunk_images = min(100, len(image_files))  # Chunk multiple images per block
        chunk_shape = (
            chunk_images,
            max_channels,
            min(chunk_shape[1], max_height),
            min(chunk_shape[2], max_width),
        )
    else:
        array_shape = (len(image_files), max_height, max_width)
        chunk_images = min(100, len(image_files))
        chunk_shape = (
            chunk_images,
            min(chunk_shape[1], max_height),
            min(chunk_shape[2], max_width),
        )

    logger.info(f"Creating Zarr array with shape {array_shape} and chunks {chunk_shape}")

    # Setup compression using Zarr v3 codecs
    compressor_map = {
        "blosc": zarr.codecs.BloscCodec,
        "zstd": zarr.codecs.ZstdCodec,
        "gzip": zarr.codecs.GzipCodec,
        "zlib": zarr.codecs.GzipCodec,  # Use gzip for zlib
        "lz4": zarr.codecs.BloscCodec,  # Use blosc with lz4
        "bz2": zarr.codecs.GzipCodec,  # Fallback to gzip
        "lzma": zarr.codecs.GzipCodec,  # Fallback to gzip
    }

    if compressor.lower() not in compressor_map:
        compressor = "blosc"  # Default fallback
        logger.warning(f"Unsupported compressor, using default: {compressor}")

    # Create appropriate codec with level optimized for speed
    if compressor.lower() in ["blosc", "lz4"]:
        # Use LZ4 for maximum speed, lower compression level
        compressor_obj = zarr.codecs.BloscCodec(
            cname="lz4", clevel=min(3, clevel), shuffle="shuffle"  # Speed over compression
        )
    elif compressor.lower() == "zstd":
        # Lower compression level for speed
        compressor_obj = zarr.codecs.ZstdCodec(level=min(3, clevel))
    else:  # gzip and others
        # Use fastest gzip level
        compressor_obj = zarr.codecs.GzipCodec(level=min(3, clevel))

    # Create Zarr store
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode="w")

    # Create the main images array
    images_array = root.create_array(
        "images",
        shape=array_shape,
        chunks=chunk_shape,
        dtype=sample_dtype,
        compressors=[compressor_obj],
        fill_value=0,
    )

    # Process images in parallel with optimized batching
    logger.info(f"Processing {len(image_files)} images with {num_parallel_workers} workers")

    # Optimize batch size for better I/O and memory usage
    # Larger batches reduce Zarr write overhead, but increase memory usage
    optimal_batch_size = max(50, min(500, len(image_files) // max(1, num_parallel_workers)))
    metadata_list = []

    # Use ThreadPoolExecutor for I/O bound operations (reading images)
    # This avoids pickle issues with zarr arrays while still providing parallelism

    with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
        futures = []

        for i in range(0, len(image_files), optimal_batch_size):
            batch = image_files[i : i + optimal_batch_size]
            future = executor.submit(_process_image_batch, batch, images_array, i, fits_extension)
            futures.append(future)

        # Collect results with progress bar
        with tqdm(total=len(futures), desc="Processing batches") as pbar:
            for future in futures:
                batch_metadata = future.result()  # Wait for completion
                metadata_list.extend(batch_metadata)
                pbar.update(1)

    # Create metadata array in Zarr
    metadata_df_images = pd.DataFrame(metadata_list)

    # Save metadata as Parquet
    parquet_path = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"

    # Merge with original metadata if possible
    if len(metadata_df_images) == len(metadata_df):
        combined_metadata = pd.concat(
            [metadata_df.reset_index(drop=True), metadata_df_images.reset_index(drop=True)], axis=1
        )
    else:
        combined_metadata = metadata_df_images

    combined_metadata.to_parquet(parquet_path)
    logger.info(f"Saved metadata to {parquet_path}")

    # Add attributes to zarr group
    root.attrs.update(
        {
            "total_images": len(image_files),
            "image_shape": array_shape[1:],
            "chunk_shape": chunk_shape[1:],
            "compressor": compressor,
            "compression_level": clevel,
            "metadata_file": str(parquet_path),
            "supported_extensions": list(I2Z_SUPPORTED_EXTS),
            "creation_info": {
                "fits_extension": fits_extension,
                "recursive_scan": recursive,
                "source_folders": [str(f) for f in folders],
            },
        }
    )

    logger.info(f"Successfully created Zarr store: {zarr_path}")
    total_size_mb = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()) / 1024**2
    logger.info(f"Total size: {total_size_mb:.2f} MB")

    return zarr_path
