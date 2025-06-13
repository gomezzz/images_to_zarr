from pathlib import Path
import ast
import zarr
import pandas as pd
from loguru import logger
import numpy as np
from collections import Counter


def inspect(store: Path | str) -> None:
    """
    Print a human-readable summary of a sharded Zarr image archive.

    The layout and wording mimic the example you supplied.

    Parameters
    ----------
    store
        Path pointing to the ``*.zarr`` directory.

    Examples
    --------
    >>> inspect_zarr(\"~/data/galaxy_cutouts.zarr\")
    ================================================================================
    SUMMARY STATISTICS
    ================================================================================
    Total images across all files: 104 857 600
    Total storage size: 126 743.31 MB
    Average file size: 126.74 MB
    File size range:  8.12 – 531.00 MB

    Format distribution:
      FITS:  60 000 000 (57.2 %)
      PNG:   30 000 000 (28.6 %)
      JPEG:  10 000 000 ( 9.5 %)
      TIFF:   4 857 600 ( 4.6 %)

    Original data type distribution:
      uint8:   78 %
      int16:   12 %
      float32: 10 %
    """
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

        # Basic statistics
        total_images = images_array.shape[0]
        image_shape = images_array.shape[1:]

        # Calculate storage sizes
        total_size_bytes = sum(f.stat().st_size for f in store_path.rglob("*") if f.is_file())
        total_size_mb = total_size_bytes / (1024**2)

        # Load metadata if available
        metadata_file = None
        parquet_files = list(store_path.parent.glob(f"{store_path.stem}_metadata.parquet"))
        if parquet_files:
            metadata_file = parquet_files[0]

        # Try to find metadata from attributes
        attrs = dict(root.attrs) if hasattr(root, "attrs") else {}

        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total images across all files: {total_images:,}")
        print(f"Total storage size: {total_size_mb:.2f} MB")
        print(f"Image dimensions: {image_shape}")
        print(f"Data type: {images_array.dtype}")

        if "compressor" in attrs:
            compression_level = attrs.get("compression_level", "unknown")
            print(f"Compression: {attrs['compressor']} (level {compression_level})")

        if "chunk_shape" in attrs:
            print(f"Chunk shape: {attrs['chunk_shape']}")

        # Calculate file size statistics if we have chunks
        try:
            chunk_files = list(store_path.rglob("*"))
            chunk_files = [
                f for f in chunk_files if f.is_file() and f.suffix not in {".json", ".zarr"}
            ]

            if chunk_files:
                chunk_sizes_mb = [f.stat().st_size / (1024**2) for f in chunk_files]
                avg_file_size = np.mean(chunk_sizes_mb)
                min_file_size = np.min(chunk_sizes_mb)
                max_file_size = np.max(chunk_sizes_mb)

                print(f"Average chunk size: {avg_file_size:.2f} MB")
                print(f"Chunk size range: {min_file_size:.2f} – {max_file_size:.2f} MB")
                print(f"Number of chunks: {len(chunk_files)}")
        except Exception as e:
            logger.debug(f"Could not analyze chunk sizes: {e}")

        # Analyze metadata if available
        if metadata_file and metadata_file.exists():
            try:
                metadata_df = pd.read_parquet(metadata_file)

                print(f"\nMetadata loaded from: {metadata_file.name}")
                print(f"Metadata records: {len(metadata_df)}")

                # Format distribution
                if "original_extension" in metadata_df.columns:
                    ext_counts = metadata_df["original_extension"].value_counts()
                    print("\nFormat distribution:")
                    for ext, count in ext_counts.items():
                        percentage = (count / total_images) * 100
                        ext_name = _format_extension_name(ext)
                        print(f"  {ext_name}: {count:,} ({percentage:.1f}%)")

                # Data type distribution
                if "dtype" in metadata_df.columns:
                    dtype_counts = metadata_df["dtype"].value_counts()
                    print("\nOriginal data type distribution:")
                    for dtype, count in dtype_counts.items():
                        percentage = (count / total_images) * 100
                        print(f"  {dtype}: {percentage:.1f}%")

                # Shape distribution
                if "shape" in metadata_df.columns:
                    # Parse shape strings back to tuples for analysis
                    shapes = []
                    for shape_str in metadata_df["shape"]:
                        try:
                            # Handle string representation of tuples
                            if isinstance(shape_str, str):
                                shape = ast.literal_eval(shape_str)
                            else:
                                shape = shape_str
                            shapes.append(shape)
                        except Exception:
                            continue

                    if shapes:
                        shape_counter = Counter(shapes)
                        print("\nImage shape distribution (top 5):")
                        for shape, count in shape_counter.most_common(5):
                            percentage = (count / len(shapes)) * 100
                            print(f"  {shape}: {count:,} ({percentage:.1f}%)")

                # File size statistics
                if "file_size_bytes" in metadata_df.columns:
                    file_sizes_mb = metadata_df["file_size_bytes"] / (1024**2)
                    print("\nOriginal file size statistics:")
                    print(f"  Mean: {file_sizes_mb.mean():.2f} MB")
                    print(f"  Median: {file_sizes_mb.median():.2f} MB")
                    print(f"  Range: {file_sizes_mb.min():.2f} – {file_sizes_mb.max():.2f} MB")

                # Dynamic range analysis
                if all(col in metadata_df.columns for col in ["min_value", "max_value"]):
                    print("\nDynamic range analysis:")
                    overall_min = metadata_df["min_value"].min()
                    overall_max = metadata_df["max_value"].max()
                    print(f"  Overall range: {overall_min:.3f} – {overall_max:.3f}")

                    if "mean_value" in metadata_df.columns:
                        mean_pixel_value = metadata_df["mean_value"].mean()
                        print(f"  Mean pixel value across all images: {mean_pixel_value:.3f}")

            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

        # Additional store information
        print("\nStore information:")
        print(f"  Store path: {store_path}")
        print(f"  Store format: Zarr v{zarr.__version__}")

        if "creation_info" in attrs:
            creation_info = attrs["creation_info"]
            print(f"  Source folders: {len(creation_info.get('source_folders', []))}")
            if "fits_extension" in creation_info:
                print(f"  FITS extension used: {creation_info['fits_extension']}")
            print(f"  Recursive scan: {creation_info.get('recursive_scan', 'unknown')}")

        if "supported_extensions" in attrs:
            extensions = attrs["supported_extensions"]
            print(f"  Supported extensions: {', '.join(extensions)}")

        print("=" * 80)

    except Exception as e:
        logger.error(f"Failed to inspect Zarr store: {e}")
        raise


def _format_extension_name(ext: str) -> str:
    """Convert file extension to readable format name."""
    ext_map = {
        ".fits": "FITS",
        ".fit": "FITS",
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }
    return ext_map.get(ext.lower(), ext.upper())
