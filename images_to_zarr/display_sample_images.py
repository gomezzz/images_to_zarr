from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from loguru import logger


def display_sample_images(
    store: Path | str,
    num_samples: int = 4,
    figsize: tuple[int, int] = (12, 8),
    save_path: Path | str | None = None,
) -> None:
    """
    Display randomly sampled images from the Zarr store.

    Parameters
    ----------
    store
        Path pointing to the ``*.zarr`` directory.
    num_samples
        Number of random images to display (default: 4).
    figsize
        Figure size as (width, height) in inches.
    save_path
        Optional path to save the figure. If None, displays interactively.

    Examples
    --------
    >>> display_sample_images("~/data/galaxy_cutouts.zarr", num_samples=6)
    >>> display_sample_images("~/data/galaxy_cutouts.zarr", save_path="samples.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib is required for image display. Install it with: pip install matplotlib"
        )
        return

    store_path = Path(store)

    if not store_path.exists():
        logger.error(f"Store does not exist: {store_path}")
        return

    if not store_path.is_dir():
        logger.error(f"Store path is not a directory: {store_path}")
        return

    try:
        # Open the Zarr store
        zarr_store = zarr.storage.LocalStore(store_path)
        root = zarr.open_group(store=zarr_store, mode="r")

        # Get the images array
        if "images" not in root:
            logger.error("No 'images' array found in the Zarr store")
            return

        images_array = root["images"]
        total_images = images_array.shape[0]

        if total_images == 0:
            logger.error("No images found in the store")
            return

        # Generate random indices
        if num_samples > total_images:
            logger.warning(
                f"Requested {num_samples} samples but only {total_images} images available"
            )
            num_samples = total_images

        indices = np.random.choice(total_images, size=num_samples, replace=False)
        indices = sorted(indices)  # Sort for better cache performance

        # Calculate subplot layout
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols

        # Create the figure
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Load metadata if available
        parquet_files = list(store_path.parent.glob(f"{store_path.stem}_metadata.parquet"))
        metadata_df = None
        if parquet_files:
            try:
                metadata_df = pd.read_parquet(parquet_files[0])
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

        # Display each sampled image
        for i, idx in enumerate(indices):
            ax = axes[i]

            # Load image data
            image_data = images_array[idx]

            # Handle different image formats (NCHW)
            if len(image_data.shape) == 3:  # CHW format
                c, h, w = image_data.shape
                if c == 1:
                    # Grayscale image
                    display_img = image_data[0]
                    cmap = "gray"
                elif c == 3:
                    # RGB image - convert from CHW to HWC
                    display_img = np.transpose(image_data, (1, 2, 0))
                    cmap = None
                else:
                    # Multi-channel - show first channel
                    display_img = image_data[0]
                    cmap = "viridis"
            elif len(image_data.shape) == 2:  # HW format
                display_img = image_data
                cmap = "gray"
            else:
                logger.warning(f"Unexpected image shape: {image_data.shape}")
                continue

            # Auto-normalize for display (handles .fits and other formats)
            display_img = _normalize_for_display(display_img)

            # Display the image
            ax.imshow(display_img, cmap=cmap, vmin=0, vmax=1, origin="lower")

            # Add title with metadata if available
            title = f"Image {idx}"
            if metadata_df is not None and idx < len(metadata_df):
                if "original_filename" in metadata_df.columns:
                    filename = metadata_df.iloc[idx]["original_filename"]
                    title = f"{idx}: {filename}"
                elif "filename" in metadata_df.columns:
                    filename = metadata_df.iloc[idx]["filename"]
                    title = f"{idx}: {filename}"

            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis("off")

        # Add overall title
        plt.suptitle(f"Random sample of {num_samples} images from {store_path.name}", fontsize=14)
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved sample images to {save_path}")
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Failed to display sample images: {e}")
        raise


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image for display, handling various data types and ranges.

    Parameters
    ----------
    image : np.ndarray
        Input image array

    Returns
    -------
    np.ndarray
        Normalized image in range [0, 1] for display
    """
    # Handle different data types
    if image.dtype == np.uint8:
        # Already in [0, 255] range
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        # In [0, 65535] range
        return image.astype(np.float32) / 65535.0
    elif image.dtype in [np.float32, np.float64]:
        # For floating point data, use robust percentile normalization
        # This handles .fits files and other scientific data well

        # Remove any NaN or infinite values
        finite_mask = np.isfinite(image)
        if not np.any(finite_mask):
            # All values are non-finite, return zeros
            return np.zeros_like(image, dtype=np.float32)

        finite_data = image[finite_mask]

        # Use 1st and 99th percentiles for robust normalization
        # This handles outliers and negative values well
        vmin, vmax = np.percentile(finite_data, [1, 99])

        # Handle edge case where vmin == vmax
        if vmax == vmin:
            # All values are the same, normalize to mid-range
            return np.full_like(image, 0.5, dtype=np.float32)

        # Normalize to [0, 1] range
        normalized = np.clip((image - vmin) / (vmax - vmin), 0, 1)
        return normalized.astype(np.float32)
    else:
        # For other integer types, convert to float and use percentile normalization
        return _normalize_for_display(image.astype(np.float32))
