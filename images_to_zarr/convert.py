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
    """Read image data from various formats."""
    file_ext = image_path.suffix.lower()
    metadata = {
        "original_filename": image_path.name,
        "original_extension": file_ext,
        "file_size_bytes": image_path.stat().st_size,
    }

    try:
        if file_ext in {".fits", ".fit"}:
            # Handle FITS files
            if fits_extension is None:
                # Try to find the first extension with data
                with fits.open(image_path) as hdul:
                    for i, hdu in enumerate(hdul):
                        if hdu.data is not None:
                            fits_extension = i
                            break
                    else:
                        raise ValueError(f"No data found in any FITS extension in {image_path}")

            if isinstance(fits_extension, (list, tuple)):
                # Concatenate multiple extensions
                arrays = []
                with fits.open(image_path) as hdul:
                    for ext in fits_extension:
                        if hdul[ext].data is not None:
                            arrays.append(hdul[ext].data)
                    metadata["fits_extensions"] = list(fits_extension)
                if not arrays:
                    raise ValueError(f"No valid data found in FITS extensions {fits_extension}")
                data = np.concatenate(arrays, axis=0 if len(arrays[0].shape) == 2 else -1)
            else:
                # Single extension
                with fits.open(image_path) as hdul:
                    data = hdul[fits_extension].data
                    if data is None:
                        raise ValueError(f"No data found in FITS extension {fits_extension}")
                    metadata["fits_extension"] = fits_extension

        else:
            # Handle other image formats (PNG, JPEG, TIFF)
            if file_ext in {".png", ".jpg", ".jpeg"}:
                # Use PIL for better format support
                with Image.open(image_path) as img:
                    data = np.array(img)
                    metadata["mode"] = img.mode
            else:
                # Use imageio for TIFF and other formats
                data = imageio.imread(image_path)

        # Ensure we have at least 2D data
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 3:
            logger.warning(f"Image {image_path} has {data.ndim} dimensions, flattening extra dims")
            data = data.reshape(data.shape[0], -1)

        metadata.update(
            {
                "dtype": str(data.dtype),
                "shape": data.shape,
                "min_value": float(np.min(data)),
                "max_value": float(np.max(data)),
                "mean_value": float(np.mean(data)),
            }
        )

        return data, metadata

    except Exception as e:
        logger.error(f"Failed to read {image_path}: {e}")
        raise


def _process_image_batch(
    image_paths: list[Path],
    zarr_array: zarr.Array,
    metadata_list: list,
    start_idx: int,
    fits_extension: int | str | Sequence[int | str] | None = None,
) -> None:
    """Process a batch of images and write to zarr array."""
    for i, image_path in enumerate(image_paths):
        try:
            data, metadata = _read_image_data(image_path, fits_extension)

            # Handle different image dimensions by padding/cropping to match zarr shape
            target_shape = zarr_array.shape[1:]  # Skip the first dimension (image index)

            if len(data.shape) == 2:
                # Add channel dimension if needed
                if len(target_shape) == 3:
                    data = data[np.newaxis, :, :]

            # Pad or crop to match target shape
            final_data = np.zeros(target_shape, dtype=data.dtype)

            # Copy data with appropriate slicing
            slices = tuple(slice(0, min(s, t)) for s, t in zip(data.shape, target_shape))
            final_data[slices] = data[slices]

            zarr_array[start_idx + i] = final_data
            metadata_list.append(metadata)

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            # Create dummy data for failed images
            dummy_data = np.zeros(zarr_array.shape[1:], dtype=zarr_array.dtype)
            zarr_array[start_idx + i] = dummy_data
            metadata_list.append(
                {
                    "original_filename": image_path.name,
                    "error": str(e),
                    "dtype": str(dummy_data.dtype),
                    "shape": dummy_data.shape,
                }
            )


def convert(
    folders: Sequence[Path] | Sequence[str],
    output_dir: Path | str,
    metadata: Path | str | None = None,
    recursive: bool = False,
    num_parallel_workers: int = 8,
    fits_extension: int | str | Sequence[int | str] | None = None,
    *,
    chunk_shape: tuple[int, int, int] = (1, 256, 256),
    shard_bytes: int = 16 * 2**20,
    compressor: str = "zstd",
    clevel: int = 4,
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
        metadata_df = pd.DataFrame({
            'filename': [img_path.name for img_path in image_files]
        })
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

    # Determine image dimensions by sampling a few files
    logger.info("Analyzing image dimensions...")
    sample_size = min(10, len(image_files))
    max_height, max_width = 0, 0
    max_channels = 1
    sample_dtype = np.uint8

    for img_path in image_files[:sample_size]:
        try:
            data, _ = _read_image_data(img_path, fits_extension)
            if len(data.shape) == 2:
                h, w = data.shape
                c = 1
            elif len(data.shape) == 3:
                c, h, w = data.shape
            else:
                continue

            max_height = max(max_height, h)
            max_width = max(max_width, w)
            max_channels = max(max_channels, c)

            # Use the most general dtype
            if np.issubdtype(data.dtype, np.floating):
                sample_dtype = np.float32
            elif data.dtype == np.uint16:
                sample_dtype = np.uint16

        except Exception as e:
            logger.warning(f"Could not analyze {img_path}: {e}")
            continue

    # Adjust chunk shape to match data dimensions
    if max_channels > 1:
        array_shape = (len(image_files), max_channels, max_height, max_width)
        chunk_shape = (1, max_channels, chunk_shape[1], chunk_shape[2])
    else:
        array_shape = (len(image_files), max_height, max_width)
        chunk_shape = (1, chunk_shape[1], chunk_shape[2])

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

    # Create appropriate codec with level
    if compressor.lower() in ["blosc", "lz4"]:
        compressor_obj = zarr.codecs.BloscCodec(
            cname="lz4" if compressor.lower() == "lz4" else "zstd", clevel=clevel
        )
    elif compressor.lower() == "zstd":
        compressor_obj = zarr.codecs.ZstdCodec(level=clevel)
    else:  # gzip and others
        compressor_obj = zarr.codecs.GzipCodec(level=clevel)

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

    # Process images in parallel
    logger.info(f"Processing {len(image_files)} images with {num_parallel_workers} workers")

    batch_size = max(1, len(image_files) // (num_parallel_workers * 4))
    metadata_list = []

    with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
        futures = []

        for i in range(0, len(image_files), batch_size):
            batch = image_files[i : i + batch_size]
            batch_metadata = []
            future = executor.submit(
                _process_image_batch, batch, images_array, batch_metadata, i, fits_extension
            )
            futures.append((future, batch_metadata))

        # Collect results with progress bar
        with tqdm(total=len(futures), desc="Processing batches") as pbar:
            for future, batch_metadata in futures:
                future.result()  # Wait for completion
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
    logger.info(
        f"Total size: {sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file()) / 1024**2:.2f} MB"
    )

    return zarr_path
