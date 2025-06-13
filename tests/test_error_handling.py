"""Test error handling and edge cases."""

import pytest
import numpy as np
import pandas as pd
import zarr

from images_to_zarr.convert import convert


class TestErrorHandling:
    """Test error handling for edge cases in new functionality."""

    def test_error_handling_edge_cases(self, temp_dir):
        """Test error handling for edge cases in new functionality."""
        output_dir = temp_dir / "output"

        # Test 1: No folders and no images provided
        with pytest.raises(ValueError, match="Must provide either folders or images"):
            convert(output_dir=output_dir)

        # Test 2: Invalid images array for direct conversion
        with pytest.raises(ValueError, match="images parameter must be a numpy array"):
            convert(images="not_an_array", output_dir=output_dir)

        # Test 3: Wrong dimensionality for direct images
        with pytest.raises(ValueError, match="Direct image input must be 4D"):
            convert(images=np.random.random((64, 64)), output_dir=output_dir)

        # Test 4: Empty folder (no images found)
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No image files found"):
            convert(folders=[empty_dir], output_dir=output_dir)

    def test_direct_memory_conversion(self, temp_dir):
        """Test conversion from numpy arrays directly."""
        # Create test images in memory
        images = np.random.randint(0, 255, (5, 1, 64, 64), dtype=np.uint8)

        # Create corresponding metadata
        metadata = [
            {
                "original_filename": f"memory_image_{i}.png",
                "dtype": str(images.dtype),
                "shape": images.shape[1:],
                "custom_field": f"value_{i}",
            }
            for i in range(len(images))
        ]

        output_dir = temp_dir / "output"
        zarr_path = convert(
            images=images,
            image_metadata=metadata,
            output_dir=output_dir,
            chunk_shape=(2, 1, 32, 32),  # Test custom chunking with correct 4D shape
            overwrite=True,
        )

        # Verify the conversion
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        assert images_array.shape == images.shape
        assert np.array_equal(images_array[:], images)
        assert root.attrs["creation_info"]["direct_memory_conversion"] is True

        # Check metadata
        metadata_path = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        assert metadata_path.exists()

        loaded_metadata = pd.read_parquet(metadata_path)
        assert len(loaded_metadata) == len(metadata)
        assert "custom_field" in loaded_metadata.columns
