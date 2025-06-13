"""Comprehensive integration tests for all features."""

import pytest
import numpy as np
import pandas as pd
import zarr
from PIL import Image

from images_to_zarr.convert import convert


class TestComprehensiveIntegration:
    """Test a comprehensive integration of all features."""

    def test_comprehensive_integration(self, temp_dir):
        """Test a comprehensive scenario with multiple features."""
        # Create a complex scenario with:
        # - Mixed image formats and sizes
        # - Custom metadata
        # - Resize functionality
        # - Custom chunking
        # - Custom compression

        images_dir = temp_dir / "complex_test"
        images_dir.mkdir()

        # Create images with different formats and sizes
        formats_and_sizes = [
            ("png", (32, 48), np.uint8),
            ("jpg", (64, 64), np.uint8),
            ("tiff", (100, 80), np.uint8),
        ]

        files = []
        metadata_entries = []

        for i, (fmt, (h, w), dtype) in enumerate(formats_and_sizes):
            # Create random image data
            if fmt == "jpg":
                # JPEG needs RGB
                data = np.random.randint(0, 255, (h, w, 3), dtype=dtype)
                mode = "RGB"
            else:
                data = np.random.randint(0, 255, (h, w), dtype=dtype)
                mode = "L"

            img_path = images_dir / f"complex_{i}.{fmt}"
            Image.fromarray(data, mode=mode).save(img_path)
            files.append(img_path)

            # Create metadata
            metadata_entries.append(
                {
                    "filename": img_path.name,
                    "object_id": f"OBJ_{i:03d}",
                    "ra": 180.0 + i * 10.0,
                    "dec": -30.0 + i * 5.0,
                    "filter": ["g", "r", "i"][i],
                    "exposure_time": [30, 60, 120][i],
                }
            )

        # Create metadata file
        metadata_df = pd.DataFrame(metadata_entries)
        metadata_path = temp_dir / "complex_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)

        # Run conversion with all features
        output_dir = temp_dir / "complex_output.zarr"
        zarr_path = convert(
            folders=[images_dir],
            metadata=metadata_path,
            output_dir=output_dir,
            resize=(50, 60),  # Resize all to same size
            interpolation_order=1,  # Bilinear
            chunk_shape=(2, 25, 30),  # Custom chunk
            compressor="zstd",
            clevel=2,
            num_parallel_workers=2,
            overwrite=True,
        )

        # Verify comprehensive results
        assert zarr_path.exists()
        assert zarr_path == output_dir  # Used exact path

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        # Check array properties
        assert images_array.shape == (3, 3, 50, 60)  # 3 images, 3 channels (RGB), resized
        assert images_array.chunks == (2, 3, 25, 30)  # Custom chunk + full channels

        # Check attributes
        attrs = root.attrs
        assert attrs["compressor"] == "zstd"
        assert attrs["compression_level"] == 2
        assert attrs["creation_info"]["resize"] == [50, 60]
        assert attrs["creation_info"]["interpolation_order"] == 1

        # Check metadata preservation
        metadata_path_out = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        combined_metadata = pd.read_parquet(metadata_path_out)

        assert len(combined_metadata) == 3
        assert "object_id" in combined_metadata.columns
        assert "filter" in combined_metadata.columns
        assert "original_filename" in combined_metadata.columns  # Added by processing

    def test_grayscale_images_same_size_regression(self, temp_dir):
        """Regression test for bug where grayscale images of same size cause shape mismatch."""
        # Create directory with multiple grayscale images of the same size
        images_dir = temp_dir / "grayscale_images"
        images_dir.mkdir()

        # Create multiple grayscale images with same dimensions
        sample_data = np.random.randint(0, 255, (424, 424), dtype=np.uint8)

        files = []
        for i in range(3):
            img_path = images_dir / f"grayscale_{i}.png"
            Image.fromarray(sample_data, mode="L").save(img_path)
            files.append(img_path)

        # This should work without errors (previously caused shape mismatch)
        zarr_path = convert(
            output_dir=temp_dir,
            folders=[images_dir],
            chunk_shape=(1, 256, 256),
            overwrite=True,
        )

        # Verify the result
        assert zarr_path.exists()

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        # Should be 3D for grayscale images: (N, H, W)
        assert images_array.shape == (3, 424, 424)
        assert images_array.ndim == 3

        # Verify data integrity
        assert np.array_equal(images_array[0], sample_data)
