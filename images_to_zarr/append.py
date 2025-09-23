"""Append functionality for adding images to existing Zarr stores."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import zarr
import pandas as pd
from loguru import logger


def append_to_zarr(
    zarr_path: Path,
    images: np.ndarray,
    metadata: pd.DataFrame | None = None,
) -> int:
    """
    Append images to an existing Zarr store.

    Parameters
    ----------
    zarr_path : Path
        Path to the existing Zarr store
    images : np.ndarray
        Images to append in NCHW format (batch, channels, height, width) or
        NCH format (batch, height, width) for grayscale
    metadata : pd.DataFrame | None
        Metadata for the new images. If None, minimal metadata will be created.

    Returns
    -------
    int
        The starting index where the new images were appended

    Raises
    ------
    FileNotFoundError
        If the Zarr store doesn't exist
    ValueError
        If the image dimensions don't match the existing store
    """
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    # Open existing store
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode="r+")

    if "images" not in root:
        raise ValueError(f"No 'images' array found in Zarr store: {zarr_path}")

    images_array = root["images"]

    # Validate image dimensions
    existing_shape = images_array.shape
    new_shape = images.shape

    # Check dimension compatibility
    if len(existing_shape) != len(new_shape):
        raise ValueError(
            f"Dimension mismatch: existing array has {len(existing_shape)} dimensions, "
            f"new images have {len(new_shape)} dimensions"
        )

    # Check that all dimensions except the first (batch) match
    if existing_shape[1:] != new_shape[1:]:
        raise ValueError(
            f"Image dimensions don't match: existing {existing_shape[1:]}, " f"new {new_shape[1:]}"
        )

    # Check dtype compatibility
    if images_array.dtype != images.dtype:
        logger.warning(
            f"Converting images from {images.dtype} to {images_array.dtype} "
            "to match existing store"
        )
        images = images.astype(images_array.dtype)

    # Get current size and resize array
    current_size = existing_shape[0]
    new_size = current_size + new_shape[0]

    logger.info(f"Appending {new_shape[0]} images to existing {current_size} images")

    # Resize the array to accommodate new images
    new_full_shape = (new_size,) + existing_shape[1:]
    images_array.resize(new_full_shape)

    # Append the new images
    images_array[current_size:new_size] = images

    # Update attributes
    attrs = dict(root.attrs)
    attrs["total_images"] = new_size

    # Track append history
    append_history = attrs.get("append_history", [])
    append_history.append(
        {
            "appended_count": new_shape[0],
            "start_index": current_size,
            "end_index": new_size,
        }
    )
    attrs["append_history"] = append_history
    root.attrs.update(attrs)

    # Handle metadata if provided
    if metadata is not None:
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"

        if metadata_parquet.exists():
            # Load existing metadata and append new
            existing_metadata = pd.read_parquet(metadata_parquet)
            combined_metadata = pd.concat([existing_metadata, metadata], ignore_index=True)
            combined_metadata.to_parquet(metadata_parquet)
            logger.info(f"Appended metadata to {metadata_parquet}")
        else:
            # Create new metadata file
            metadata.to_parquet(metadata_parquet)
            logger.info(f"Created new metadata file: {metadata_parquet}")

    logger.info(f"Successfully appended {new_shape[0]} images to {zarr_path}")
    logger.info(f"New total: {new_size} images")

    return current_size


def validate_append_compatibility(
    zarr_path: Path,
    new_images_shape: tuple,
    new_images_dtype: np.dtype,
) -> bool:
    """
    Check if new images are compatible for appending to existing store.

    Parameters
    ----------
    zarr_path : Path
        Path to the existing Zarr store
    new_images_shape : tuple
        Shape of the new images to append
    new_images_dtype : np.dtype
        Data type of the new images

    Returns
    -------
    bool
        True if compatible, raises ValueError otherwise
    """
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode="r")

    if "images" not in root:
        raise ValueError(f"No 'images' array found in Zarr store: {zarr_path}")

    images_array = root["images"]
    existing_shape = images_array.shape

    # Check dimensions match (except batch dimension)
    if len(existing_shape) != len(new_images_shape):
        raise ValueError(
            f"Dimension mismatch: existing has {len(existing_shape)} dimensions, "
            f"new has {len(new_images_shape)} dimensions"
        )

    if existing_shape[1:] != new_images_shape[1:]:
        raise ValueError(
            f"Image dimensions don't match: existing {existing_shape[1:]}, "
            f"new {new_images_shape[1:]}"
        )

    # Warn about dtype conversion
    if images_array.dtype != new_images_dtype:
        logger.warning(
            f"Data type mismatch: existing {images_array.dtype}, "
            f"new {new_images_dtype}. Conversion will be performed."
        )

    return True
