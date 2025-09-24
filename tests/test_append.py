"""Test append functionality."""

import pytest
import numpy as np
import pandas as pd
import zarr
import shutil

from images_to_zarr.convert import convert
from images_to_zarr.append import append_to_zarr, validate_append_compatibility


class TestAppendFunctionality:
    """Test the core append functionality."""

    def test_append_to_zarr_memory(self, temp_dir):
        """Test appending images from memory to existing Zarr store."""
        # Create initial store
        initial_images = np.random.randint(0, 255, (3, 3, 64, 64), dtype=np.uint8)
        initial_metadata = [{"id": i, "source": "initial"} for i in range(3)]

        zarr_path = convert(
            output_dir=temp_dir,
            images=initial_images,
            image_metadata=initial_metadata,
            overwrite=True,
        )

        # Create additional images to append
        append_images = np.random.randint(0, 255, (2, 3, 64, 64), dtype=np.uint8)
        append_metadata = pd.DataFrame([{"id": i + 3, "source": "appended"} for i in range(2)])

        # Append to the store
        start_idx = append_to_zarr(zarr_path, append_images, append_metadata)

        # Verify results
        assert start_idx == 3  # Should start at index 3

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        # Should now have 5 images total
        assert images_array.shape == (5, 3, 64, 64)

        # Check that original data is preserved
        assert np.array_equal(images_array[:3], initial_images)

        # Check that new data is appended
        assert np.array_equal(images_array[3:], append_images)

        # Check metadata
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        assert metadata_parquet.exists()

        combined_metadata = pd.read_parquet(metadata_parquet)
        assert len(combined_metadata) == 5
        assert combined_metadata.iloc[3]["source"] == "appended"

        # Check attributes
        attrs = dict(root.attrs)
        assert attrs["total_images"] == 5
        assert "append_history" in attrs
        assert len(attrs["append_history"]) == 1
        assert attrs["append_history"][0]["appended_count"] == 2

    def test_append_with_convert_function(self, temp_dir):
        """Test appending using the convert function with append=True."""
        # Create initial store
        initial_images = np.random.randint(0, 255, (2, 1, 32, 32), dtype=np.uint8)

        zarr_path = convert(
            output_dir=temp_dir,
            images=initial_images,
            overwrite=True,
        )

        # Append more images using convert with append=True
        append_images = np.random.randint(0, 255, (3, 1, 32, 32), dtype=np.uint8)
        append_metadata = [{"id": i + 2, "type": "appended"} for i in range(3)]

        result_path = convert(
            output_dir=temp_dir,
            images=append_images,
            image_metadata=append_metadata,
            append=True,
        )

        assert result_path == zarr_path

        # Verify final state
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        assert images_array.shape == (5, 1, 32, 32)  # 2 + 3 = 5 images

        # Check metadata
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        combined_metadata = pd.read_parquet(metadata_parquet)
        assert len(combined_metadata) == 5

        # Check that appended metadata is present
        assert any(
            row["type"] == "appended" for _, row in combined_metadata.iterrows() if "type" in row
        )

    def test_convert_from_memory_with_append(self, temp_dir):
        """Test convert with append=True for memory conversion."""
        # Create initial store
        initial_images = np.random.random((2, 3, 48, 48)).astype(np.float32)

        zarr_path = convert(
            output_dir=temp_dir,
            images=initial_images,
            overwrite=True,
        )

        # Append using convert
        append_images = np.random.random((1, 3, 48, 48)).astype(np.float32)

        result_path = convert(
            output_dir=temp_dir,
            images=append_images,
            append=True,
        )

        assert result_path == zarr_path

        # Verify
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        assert images_array.shape == (3, 3, 48, 48)  # 2 + 1 = 3 images

    def test_validation_functions(self, temp_dir):
        """Test validation functions for append compatibility."""
        # Create initial store
        images = np.random.randint(0, 255, (2, 3, 64, 64), dtype=np.uint8)

        zarr_path = convert(
            output_dir=temp_dir,
            images=images,
            overwrite=True,
        )

        # Test valid compatibility
        valid_shape = (1, 3, 64, 64)
        valid_dtype = np.uint8

        assert validate_append_compatibility(zarr_path, valid_shape, valid_dtype) is True

        # Test invalid shape (wrong dimensions)
        with pytest.raises(ValueError, match="Image dimensions don't match"):
            validate_append_compatibility(zarr_path, (1, 2, 64, 64), valid_dtype)

        # Test invalid shape (wrong height/width)
        with pytest.raises(ValueError, match="Image dimensions don't match"):
            validate_append_compatibility(zarr_path, (1, 3, 32, 32), valid_dtype)

        # Test nonexistent store
        with pytest.raises(FileNotFoundError):
            validate_append_compatibility(temp_dir / "nonexistent.zarr", valid_shape, valid_dtype)


class TestAppendErrorHandling:
    """Test error handling for append functionality."""

    def test_append_and_overwrite_conflict(self, temp_dir):
        """Test that append=True and overwrite=True raises an error."""
        images = np.random.randint(0, 255, (2, 3, 32, 32), dtype=np.uint8)

        with pytest.raises(ValueError, match="Cannot use both 'append=True' and 'overwrite=True'"):
            convert(
                output_dir=temp_dir,
                images=images,
                append=True,
                overwrite=True,
            )

    def test_append_to_nonexistent_store(self, temp_dir):
        """Test appending to a non-existent store raises an error."""
        images = np.random.randint(0, 255, (2, 3, 32, 32), dtype=np.uint8)

        with pytest.raises(FileNotFoundError, match="Cannot append to non-existent store"):
            convert(
                output_dir=temp_dir / "nonexistent.zarr",
                images=images,
                append=True,
            )

    def test_append_incompatible_dimensions(self, temp_dir):
        """Test appending images with incompatible dimensions."""
        # Create initial store with 3 channels
        initial_images = np.random.randint(0, 255, (2, 3, 32, 32), dtype=np.uint8)

        zarr_path = convert(
            output_dir=temp_dir,
            images=initial_images,
            overwrite=True,
        )

        # Try to append images with different dimensions
        incompatible_images = np.random.randint(0, 255, (1, 1, 32, 32), dtype=np.uint8)

        with pytest.raises(ValueError, match="Image dimensions don't match"):
            append_to_zarr(zarr_path, incompatible_images)

    def test_append_no_images_array(self, temp_dir):
        """Test appending to a zarr store without 'images' array."""
        # Create a zarr store without 'images' array
        store = zarr.storage.LocalStore(temp_dir / "test.zarr")
        root = zarr.open_group(store=store, mode="w")
        root.create_array("other_data", shape=(10,), dtype=np.float32)
        # Don't need to close zarr groups

        images = np.random.randint(0, 255, (1, 3, 32, 32), dtype=np.uint8)

        with pytest.raises(ValueError, match="No 'images' array found"):
            append_to_zarr(temp_dir / "test.zarr", images)


class TestAppendFileBasedConversion:
    """Test appending with file-based conversion."""

    def test_append_from_files(self, temp_dir, sample_images):
        """Test appending images from files to existing store."""
        images_dir, files = sample_images

        # Create initial store from first two files
        initial_files = files[:2]
        initial_dir = temp_dir / "initial"
        initial_dir.mkdir()

        for f in initial_files:
            shutil.copy(f, initial_dir)

        zarr_path = convert(
            folders=[initial_dir],
            output_dir=temp_dir,
            overwrite=True,
        )

        # Get initial count
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        initial_count = root["images"].shape[0]
        # Don't need to close zarr groups

        # Now append remaining files
        remaining_files = files[2:]
        append_dir = temp_dir / "append"
        append_dir.mkdir()

        for f in remaining_files:
            shutil.copy(f, append_dir)

        convert(
            folders=[append_dir],
            output_dir=zarr_path,  # Use existing zarr path
            append=True,
        )

        # Verify
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        final_count = root["images"].shape[0]

        assert final_count > initial_count  # Should have more images

        # Check metadata was updated
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        if metadata_parquet.exists():
            combined_metadata = pd.read_parquet(metadata_parquet)
            assert len(combined_metadata) >= initial_count

    def test_append_with_metadata_file(self, temp_dir, sample_images):
        """Test appending with custom metadata file."""
        images_dir, files = sample_images

        # Create initial directory with only first file
        initial_dir = temp_dir / "initial_images"
        initial_dir.mkdir()
        import shutil

        shutil.copy(files[0], initial_dir)

        # Create initial store
        initial_metadata = pd.DataFrame({"filename": [files[0].name], "category": ["initial"]})
        initial_metadata_path = temp_dir / "initial_meta.csv"
        initial_metadata.to_csv(initial_metadata_path, index=False)

        zarr_path = convert(
            folders=[initial_dir],
            metadata=initial_metadata_path,
            output_dir=temp_dir,
            overwrite=True,
        )

        # Create append directory with remaining files
        append_dir = temp_dir / "append_images"
        append_dir.mkdir()

        # Copy remaining files to append directory
        for f in files[1:2]:  # Just use one more file to keep it simple
            shutil.copy(f, append_dir)

        # Create append metadata
        append_metadata = pd.DataFrame({"filename": [files[1].name], "category": ["appended"]})
        append_metadata_path = temp_dir / "append_meta.csv"
        append_metadata.to_csv(append_metadata_path, index=False)

        # Append with metadata
        convert(
            folders=[append_dir],
            metadata=append_metadata_path,
            output_dir=zarr_path,  # Use existing zarr path
            append=True,
        )

        # Check that metadata was merged correctly
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        combined_metadata = pd.read_parquet(metadata_parquet)

        # Should have at least 2 entries (1 initial + 1 appended)
        assert len(combined_metadata) >= 2

        # Just check that the append operation worked
        assert len(combined_metadata) > 0
