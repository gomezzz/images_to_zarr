from __future__ import annotations
from pathlib import Path
from typing import Sequence
import pandas as pd
import numpy as np
import zarr
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from loguru import logger
from astropy.io import fits
from PIL import Image
from skimage import transform

from images_to_zarr import I2Z_SUPPORTED_EXTS
from images_to_zarr.append import append_to_zarr


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


def _normalize_folders_input(
    folders: Sequence[Path] | Sequence[str] | Path | str | None,
) -> list[Path]:
    """Normalize folders input to always be a list of Path objects."""
    if folders is None:
        return []

    if isinstance(folders, (str, Path)):
        return [Path(folders)]

    return [Path(folder) for folder in folders]


def _read_image_data(
    image_path: Path, fits_extension: int | str | Sequence[int | str] | None = None
) -> tuple[np.ndarray, dict]:
    """Read image data from various formats and return raw data with metadata."""
    file_ext = image_path.suffix.lower()

    # Essential metadata for tests and functionality
    metadata = {
        "original_filename": image_path.name,
        "original_extension": file_ext,
    }

    try:
        if file_ext in {".fits", ".fit"}:
            # Handle FITS files
            with fits.open(image_path) as hdul:
                if fits_extension is None:
                    # Find first extension with data
                    fits_extension = 0
                    for i, hdu in enumerate(hdul):
                        if hdu.data is not None:
                            fits_extension = i
                            break

                if isinstance(fits_extension, (list, tuple)):
                    # Concatenate multiple extensions along channel axis
                    arrays = []
                    for ext in fits_extension:
                        if hdul[ext].data is not None:
                            arrays.append(hdul[ext].data)
                    if not arrays:
                        raise ValueError(f"No valid data found in FITS extensions {fits_extension}")
                    # Stack along first axis (channels)
                    data = np.stack(arrays, axis=0)
                    metadata["fits_extensions"] = list(fits_extension)
                else:
                    data = hdul[fits_extension].data
                    if data is None:
                        raise ValueError(f"No data found in FITS extension {fits_extension}")
                    metadata["fits_extension"] = fits_extension

        else:
            # Handle other image formats
            if file_ext in {".png", ".jpg", ".jpeg"}:
                # Use PIL for better format support
                with Image.open(image_path) as img:
                    data = np.array(img)
                    metadata["mode"] = img.mode
            else:
                # Use imageio for TIFF and other formats
                import imageio

                data = imageio.imread(image_path)

        # Essential metadata for functionality (store raw data shape/dtype)
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


def _ensure_nchw_format(data: np.ndarray) -> np.ndarray:
    """Ensure image data is in NCHW format (batch, channels, height, width)."""
    if data.ndim == 1:
        # 1D array - reshape to (1, 1, 1, W)
        data = data.reshape(1, 1, 1, -1)
    elif data.ndim == 2:
        # 2D array (H, W) -> (1, 1, H, W)
        data = data[np.newaxis, np.newaxis, :, :]
    elif data.ndim == 3:
        # Check if it's HWC or CHW format
        h, w, c = data.shape
        if c <= 4:  # Likely HWC format (channels last)
            # Convert HWC to NCHW: (H, W, C) -> (1, C, H, W)
            data = np.transpose(data, (2, 0, 1))[np.newaxis, :, :, :]
        else:
            # Likely CHW format already or unusual shape - assume CHW
            # Convert CHW to NCHW: (C, H, W) -> (1, C, H, W)
            data = data[np.newaxis, :, :, :]
    elif data.ndim == 4:
        # Already in batch format - assume NCHW
        pass
    else:
        logger.warning(f"Image has {data.ndim} dimensions, flattening extra dims")
        # Flatten to 2D and then convert
        data = data.reshape(data.shape[0], -1)
        data = data[np.newaxis, np.newaxis, :, :]

    return data


def _resize_image(
    data: np.ndarray, new_size: tuple[int, int], interpolation_order: int = 1
) -> np.ndarray:
    """Resize image using scikit-image with specified interpolation order."""

    if data.ndim == 2:
        # Grayscale image (H, W)
        return transform.resize(
            data, new_size, order=interpolation_order, preserve_range=True, anti_aliasing=None
        ).astype(data.dtype)
    elif data.ndim == 3:
        # Multi-channel image (C, H, W) or (H, W, C)
        if data.shape[0] <= 4:  # Assume CHW format
            resized_channels = []
            for c in range(data.shape[0]):
                resized_channel = transform.resize(
                    data[c],
                    new_size,
                    order=interpolation_order,
                    preserve_range=True,
                    anti_aliasing=None,
                )
                resized_channels.append(resized_channel)
            return np.stack(resized_channels, axis=0).astype(data.dtype)
        else:  # Assume HWC format
            return transform.resize(
                data,
                new_size + (data.shape[2],),
                order=interpolation_order,
                preserve_range=True,
                anti_aliasing=None,
            ).astype(data.dtype)
    else:
        raise ValueError(f"Cannot resize image with {data.ndim} dimensions")


def _process_single_image(
    image_path: Path,
    target_shape: tuple,
    target_dtype: np.dtype,
    fits_extension: int | str | Sequence[int | str] | None = None,
    resize: tuple[int, int] | None = None,
    interpolation_order: int = 1,
) -> tuple[np.ndarray, dict]:
    """Process a single image efficiently."""
    try:
        # Read raw image data
        data, metadata = _read_image_data(image_path, fits_extension)
        # Convert to NCHW format
        data = _ensure_nchw_format(data)

        # Remove batch dimension for single images (squeeze only axis 0)
        if data.ndim == 4 and data.shape[0] == 1:
            data = data.squeeze(axis=0)  # (1, C, H, W) -> (C, H, W)

        # For single-channel images, also remove the channel dimension if target is 2D
        if data.shape[0] == 1 and len(target_shape) == 2:
            data = data.squeeze(axis=0)  # (1, H, W) -> (H, W)

        # Apply resizing if requested
        if resize is not None:
            height, width = resize
            if data.ndim == 3:  # CHW format
                if data.shape[1:] != (height, width):
                    data = _resize_image(data, (height, width), interpolation_order)
            elif data.ndim == 2:  # HW format
                if data.shape != (height, width):
                    data = _resize_image(data, (height, width), interpolation_order)
        else:
            # Check if dimensions match target - raise error if not
            if data.shape != target_shape:
                raise ValueError(
                    f"Image {image_path.name} has shape {data.shape} but expected {target_shape}. "
                    "Use the 'resize' parameter to automatically resize images."
                )

        # Ensure final data matches target shape
        if data.shape != target_shape:
            # Efficient resize/crop without creating full zeros array
            final_data = np.zeros(target_shape, dtype=target_dtype)
            # Copy data with appropriate slicing
            slices = tuple(slice(0, min(s, t)) for s, t in zip(data.shape, target_shape))
            final_data[slices] = data[slices]
            data = final_data

        # Update metadata to reflect final processed shape and dtype
        metadata.update(
            {
                "processed_shape": data.shape,
                "processed_dtype": str(target_dtype),
            }
        )

        return data.astype(target_dtype), metadata

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


def _process_image_batch(
    image_paths: list[Path],
    zarr_array: zarr.Array,
    start_idx: int,
    fits_extension: int | str | Sequence[int | str] | None = None,
    resize: tuple[int, int] | None = None,
    interpolation_order: int = 1,
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
            image_path, target_shape, target_dtype, fits_extension, resize, interpolation_order
        )
        batch_data[i] = data
        batch_metadata.append(metadata)

    # Single batch write to Zarr (much more efficient)
    zarr_array[start_idx : start_idx + batch_size] = batch_data

    return batch_metadata


def convert(
    output_dir: Path | str,
    folders: Sequence[Path] | Sequence[str] | Path | str | None = None,
    metadata: Path | str | None = None,
    recursive: bool = False,
    num_parallel_workers: int = 8,
    fits_extension: int | str | Sequence[int | str] | None = None,
    resize: tuple[int, int] | None = None,
    interpolation_order: int = 1,
    *,
    images: np.ndarray | None = None,
    image_metadata: list[dict] | None = None,
    chunk_shape: tuple[int, int, int] = (1, 256, 256),
    compressor: str = "lz4",
    clevel: int = 1,
    overwrite: bool = False,
    append: bool = False,
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
        If not provided, metadata will be created from just the filenames.    output_dir
        Destination path. If the path ends with '.zarr', it will be used as the
        zarr store path directly. Otherwise, a directory called ``<name>.zarr``
        is created inside it. Existing stores are refused unless *overwrite* is set.
    num_parallel_workers
        Threads or processes used to ingest images and write chunks.
    fits_extension
        Which FITS HDU(s) to read:

        * ``None``  →  use extension 0
        * *int* or *str*  →  single HDU
        * *Sequence*  →  concatenate multiple HDUs along the channel axis
    resize
        If specified as (height, width), all images will be resized to this size.
        If None, images must have matching dimensions or an error is raised.
    interpolation_order
        Interpolation order for resizing (0-5):

        * 0: Nearest-neighbor
        * 1: Bi-linear (default)
        * 2: Bi-quadratic
        * 3: Bi-cubic
        * 4: Bi-quartic
        * 5: Bi-quintic
    chunk_shape
        Chunk layout **(n_images, height, width)** ; the first dimension
        **must be 1** so each image maps to exactly one chunk.
    compressor
        Any *numcodecs* codec name (``"zstd"``, ``"lz4"``, …).
    clevel
        Compression level handed to *numcodecs*.
    overwrite
        Destroy an existing store at *output_dir* if present.
    append
        Append images to an existing Zarr store. Cannot be used with overwrite=True.

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

    # Validate append and overwrite parameters
    if append and overwrite:
        raise ValueError("Cannot use both 'append=True' and 'overwrite=True'")

    # Validate interpolation order
    if not (0 <= interpolation_order <= 5):
        raise ValueError("interpolation_order must be between 0 and 5")

    # Convert inputs to Path objects
    output_dir = Path(output_dir)

    # Validate that we have either folders or images
    if folders is None and images is None:
        raise ValueError("Must provide either folders or images")

    # Handle direct memory conversion vs file-based conversion
    if images is not None:
        # Direct memory conversion path
        if not isinstance(images, np.ndarray):
            raise ValueError("images parameter must be a numpy array")

        # Validate that images are 4D for direct conversion
        # (but we'll auto-convert in _ensure_nchw_format)
        if images.ndim != 4:
            raise ValueError("Direct image input must be 4D (NCHW format)")

        # Ensure NCHW format for direct images (should already be 4D but let's be safe)
        images = _ensure_nchw_format(images)

        # Use provided metadata or create default
        if image_metadata is None:
            image_metadata = [
                {
                    "original_filename": f"image_{i}.png",
                    "dtype": str(images.dtype),
                    "shape": images.shape[1:],
                }
                for i in range(len(images))
            ]

        # Set image_files to empty list since we're using direct images
        image_files = []
        store_name = "images.zarr"
    else:
        # File-based conversion path
        # Normalize folders input
        folders = _normalize_folders_input(folders)

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

    # Handle output path: if user provided a .zarr path, use it directly
    # Otherwise create a .zarr directory inside the provided directory
    if str(output_dir).endswith(".zarr"):
        zarr_path = output_dir
    else:
        zarr_path = output_dir / store_name

    if zarr_path.exists():
        if append:
            # Validate compatibility and append mode
            logger.info(f"Appending to existing store: {zarr_path}")
        elif overwrite:
            import shutil

            shutil.rmtree(zarr_path)
            logger.info(f"Removed existing store: {zarr_path}")
        else:
            raise FileExistsError(f"Store already exists: {zarr_path}")
    elif append:
        raise FileNotFoundError(f"Cannot append to non-existent store: {zarr_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine image dimensions
    logger.info("Analyzing image dimensions...")

    if images is not None:
        # Direct memory conversion - get dimensions from provided array
        if images.ndim != 4:
            raise ValueError(f"Direct images array must be 4D (NCHW), got shape {images.shape}")

        num_images, max_channels, max_height, max_width = images.shape
        sample_dtype = images.dtype

        # Create metadata DataFrame from provided metadata
        metadata_df = pd.DataFrame(image_metadata)

    else:
        # File-based conversion - determine dimensions
        num_images = len(image_files)

        # Sample files to determine dimensions, channels, and dtype
        sample_size = min(3, len(image_files))
        max_channels = 1
        sample_dtype = np.uint8
        detected_height, detected_width = None, None
        # Analyze sample images to determine properties
        for img_path in image_files[:sample_size]:
            try:
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        # Properly detect channel count based on mode
                        if img.mode == "RGB":
                            c = 3
                        elif img.mode == "RGBA":
                            c = 4
                        elif img.mode in ["L", "P"]:  # Grayscale, Palette
                            c = 1
                        elif img.mode in ["LA"]:  # Grayscale + Alpha
                            c = 2
                        else:
                            # Fallback for other modes
                            c = len(img.getbands()) if hasattr(img, "getbands") else 1

                        if img.mode in ["I", "I;16"]:
                            sample_dtype = np.uint16
                        elif img.mode == "F":
                            sample_dtype = np.float32
                            c = 1
                        elif img.mode in ["LA"]:  # Grayscale + Alpha
                            c = 2
                        else:
                            # Fallback for other modes
                            c = len(img.getbands()) if hasattr(img, "getbands") else 1

                        if img.mode in ["I", "I;16"]:
                            sample_dtype = np.uint16
                        elif img.mode == "F":
                            sample_dtype = np.float32
                else:
                    data, _ = _read_image_data(img_path, fits_extension)
                    data_nchw = _ensure_nchw_format(data)

                    if len(data_nchw.shape) == 4:
                        _, c, h, w = data_nchw.shape
                    elif len(data.shape) == 2:
                        h, w = data.shape
                        c = 1
                    elif len(data.shape) == 3:
                        if data.shape[2] <= 4:  # HWC format
                            h, w, c = data.shape
                        else:  # CHW format
                            c, h, w = data.shape

                    if np.issubdtype(data.dtype, np.floating):
                        sample_dtype = np.float32
                    elif data.dtype == np.uint16:
                        sample_dtype = np.uint16

                # Track max channels across all images
                max_channels = max(max_channels, c)
                # Track dimensions for validation (if no resize)
                if detected_height is None:
                    detected_height, detected_width = h, w
                elif (h, w) != (detected_height, detected_width) and resize is None:
                    raise ValueError(
                        f"Image {img_path.name} has dimensions {h}x{w} but expected "
                        f"{detected_height}x{detected_width}. All images must have the same "
                        "dimensions or use the 'resize' parameter to automatically resize them."
                    )

            except ValueError as ve:
                # Re-raise ValueError (dimension mismatch) immediately
                if "All images must have the same dimensions" in str(ve):
                    raise ve
                else:
                    logger.warning(f"Could not analyze {img_path}: {ve}")
                    continue
            except Exception as e:
                logger.warning(f"Could not analyze {img_path}: {e}")
                continue

        # Determine final dimensions
        if resize is not None:
            # User specified resize - use those dimensions
            max_height, max_width = resize
            logger.info(f"Using resize dimensions: {max_height}x{max_width}")
        else:
            # No resize - use detected dimensions (all must match)
            if detected_height is None or detected_width is None:
                raise ValueError("Could not determine image dimensions from sample files")
            max_height, max_width = detected_height, detected_width
            logger.info(f"Using detected dimensions: {max_height}x{max_width}")

        # Validate that we have valid dimensions
        if max_height is None or max_width is None:
            raise ValueError(
                "Could not determine image dimensions"
            )  # Determine array shape based on channels
    if images is not None:
        # For memory conversion, always preserve the input shape (always 4D)
        array_shape = images.shape
    elif max_channels > 1:
        array_shape = (num_images, max_channels, max_height, max_width)
    else:
        array_shape = (num_images, max_height, max_width)

    # Handle chunk_shape: respect user input or create smart defaults
    if len(chunk_shape) == len(array_shape):
        # User chunk shape matches array dimensions - use it (clamp to array size)
        final_chunk_shape = tuple(min(c, s) for c, s in zip(chunk_shape, array_shape))
        logger.info(f"Using user-specified chunk shape: {final_chunk_shape}")
    elif len(chunk_shape) == 3 and len(array_shape) == 4:
        # User provided 3D chunk shape but we have 4D array - expand to include channels
        final_chunk_shape = (
            min(chunk_shape[0], num_images),
            max_channels,
            min(chunk_shape[1], max_height),
            min(chunk_shape[2], max_width),
        )
        logger.info(f"Expanded 3D chunk shape to 4D: {final_chunk_shape}")
    else:
        # Create smart defaults
        default_chunk_size = min(10, num_images)  # Reasonable default
        if max_channels > 1:
            final_chunk_shape = (
                default_chunk_size,
                max_channels,
                min(256, max_height),
                min(256, max_width),
            )
        else:
            final_chunk_shape = (
                default_chunk_size,
                min(256, max_height),
                min(256, max_width),
            )
        logger.info(f"Using default chunk shape: {final_chunk_shape}")

    chunk_shape = final_chunk_shape

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

    # Handle append mode vs create mode
    if append:
        # In append mode, we just prepare the data and call append_to_zarr
        if images is not None:
            # Direct memory append
            logger.info("Appending images from memory to existing Zarr...")

            # Prepare metadata DataFrame
            if image_metadata:
                metadata_df_append = pd.DataFrame(image_metadata)
            else:
                metadata_df_append = None

            # Call append function
            append_to_zarr(zarr_path, images, metadata_df_append)

            return zarr_path
        else:
            # File-based append - need to process images first
            # We'll process them and then append
            pass  # Continue with normal flow but append at the end

    if not append:
        # Create new Zarr store
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
    else:
        # Open existing store for append
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r+")
        images_array = root["images"]

    # Process images in parallel with optimized batching
    logger.info(f"Processing {num_images} images with {num_parallel_workers} workers")

    metadata_list = []

    if images is not None and not append:
        # Direct memory conversion - write directly to zarr
        logger.info("Writing images from memory to Zarr...")

        # Write images directly to zarr array
        images_array[:] = images

        # Use provided metadata
        metadata_list = image_metadata

    elif images is None:
        # File-based conversion with parallel processing
        # Optimize batch size for better I/O and memory usage
        # Larger batches reduce Zarr write overhead, but increase memory usage
        optimal_batch_size = max(50, min(500, num_images // max(1, num_parallel_workers)))

        # Use ThreadPoolExecutor for I/O bound operations (reading images)
        # This avoids pickle issues with zarr arrays while still providing parallelism

        with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
            futures = []

            for i in range(0, len(image_files), optimal_batch_size):
                batch = image_files[i : i + optimal_batch_size]
                future = executor.submit(
                    _process_image_batch,
                    batch,
                    images_array,
                    i,
                    fits_extension,
                    resize,
                    interpolation_order,
                )
                futures.append(future)

            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Processing batches") as pbar:
                for future in futures:
                    batch_metadata = future.result()  # Wait for completion
                    metadata_list.extend(batch_metadata)
                    pbar.update(1)

    # Handle append mode for file-based conversion
    if append and images is None:
        # Process images to numpy array first
        logger.info("Processing images for append...")

        # Create a temporary array to hold all processed images
        processed_images = np.zeros((len(image_files),) + array_shape[1:], dtype=sample_dtype)

        # Process all images
        for i, img_path in enumerate(image_files):
            data, img_metadata = _process_single_image(
                img_path, array_shape[1:], sample_dtype, fits_extension, resize, interpolation_order
            )
            processed_images[i] = data
            metadata_list.append(img_metadata)

        # Create metadata DataFrame
        metadata_df_images = pd.DataFrame(metadata_list)

        # Merge with original metadata if possible
        if metadata is not None and len(metadata_df_images) == len(metadata_df):
            combined_metadata = pd.concat(
                [metadata_df.reset_index(drop=True), metadata_df_images.reset_index(drop=True)],
                axis=1,
            )
        else:
            combined_metadata = metadata_df_images

        # Append to existing store
        append_to_zarr(zarr_path, processed_images, combined_metadata)

    elif not append:
        # Normal mode - create new store
        # Create metadata array in Zarr
        metadata_df_images = pd.DataFrame(metadata_list)

        # Save metadata as Parquet
        parquet_path = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"

        # Merge with original metadata if possible
        if images is not None:
            # For direct memory conversion, use the provided metadata
            combined_metadata = metadata_df_images
        elif len(metadata_df_images) == len(metadata_df):
            combined_metadata = pd.concat(
                [metadata_df.reset_index(drop=True), metadata_df_images.reset_index(drop=True)],
                axis=1,
            )
        else:
            combined_metadata = metadata_df_images

        combined_metadata.to_parquet(parquet_path)
        logger.info(f"Saved metadata to {parquet_path}")

        # Add attributes to zarr group
        root.attrs.update(
            {
                "total_images": num_images,
                "image_shape": array_shape[1:],
                "chunk_shape": chunk_shape[1:],
                "compressor": compressor,
                "compression_level": clevel,
                "metadata_file": str(parquet_path),
                "supported_extensions": list(I2Z_SUPPORTED_EXTS),
                "creation_info": {
                    "fits_extension": fits_extension,
                    "recursive_scan": recursive,
                    "source_folders": [str(f) for f in folders] if folders else [],
                    "direct_memory_conversion": images is not None,
                    "resize": resize,
                    "interpolation_order": interpolation_order,
                },
            }
        )

    logger.info(f"Successfully created Zarr store: {zarr_path}")
    total_size_mb = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()) / 1024**2
    logger.info(f"Total size: {total_size_mb:.2f} MB")

    return zarr_path
