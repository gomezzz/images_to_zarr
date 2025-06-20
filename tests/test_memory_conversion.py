"""Test direct memory conversion and path structure."""

import pytest
import numpy as np
import zarr
import shutil

from images_to_zarr.convert import convert


class TestDirectImageConversion:
    """Test converting images directly from memory."""

    def test_convert_nchw(self, temp_dir):
        """Test converting images directly from memory in NCHW format."""
        # Create sample images in NCHW format
        batch_size = 5
        channels = 3
        height = 64
        width = 64

        images = np.random.randint(0, 255, (batch_size, channels, height, width), dtype=np.uint8)

        # Create metadata
        metadata = [{"filename": f"memory_image_{i}.unknown", "id": i} for i in range(batch_size)]

        # Convert from memory
        zarr_path = convert(
            output_dir=temp_dir,
            images=images,
            image_metadata=metadata,
            overwrite=True,
        )

        # Check the result
        assert zarr_path.exists()
        assert zarr_path.is_dir()

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        # Should have the same shape and data
        assert images_array.shape == images.shape
        assert np.array_equal(images_array[:], images)

    def test_convert_with_convenience_function(self, temp_dir):
        """Test the convenience function convert."""

        # Create sample images
        images = np.random.random((3, 2, 32, 32)).astype(np.float32)

        zarr_path = convert(
            images=images,
            output_dir=temp_dir,
            overwrite=True,
        )

        assert zarr_path.exists()

        # Check that the path structure is correct
        assert zarr_path.name == "images.zarr"
        assert zarr_path.parent == temp_dir

        # Check that there's no nested images.zarr/images.zarr
        nested_zarr = zarr_path / "images.zarr"
        assert not nested_zarr.exists(), f"Found nested zarr structure: {nested_zarr}"

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        assert np.allclose(images_array[:], images)

    def test_invalid_direct_image_input(self, temp_dir):
        """Test that invalid direct image input raises appropriate errors."""
        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Direct image input must be 4D"):
            convert(
                output_dir=temp_dir,
                images=np.random.random((64, 64)),  # 2D instead of 4D
                overwrite=True,
            )

        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Direct image input must be 4D"):
            convert(
                output_dir=temp_dir,
                images=np.random.random((5, 64, 64)),  # 3D instead of 4D
                overwrite=True,
            )

    def test_no_folders_and_no_images_error(self, temp_dir):
        """Test that providing neither folders nor images raises an error."""
        with pytest.raises(ValueError, match="Must provide either folders or images"):
            convert(
                output_dir=temp_dir,
                folders=None,
                images=None,
                overwrite=True,
            )


class TestPathStructure:
    """Test that zarr paths are created correctly for both folder and memory conversion."""

    def test_path_structure_correctness(self, temp_dir, sample_images):
        """Test that zarr paths are created correctly for both folder and memory conversion."""
        # Test folder-based conversion
        images_dir, files = sample_images

        zarr_path_folder = convert(
            folders=[images_dir],
            output_dir=temp_dir,
            overwrite=True,
        )

        assert zarr_path_folder.exists()
        assert zarr_path_folder.name == "images.zarr"
        assert zarr_path_folder.parent == temp_dir

        # Check that there's no nested zarr structure
        nested_zarr_folder = zarr_path_folder / "images.zarr"
        assert (
            not nested_zarr_folder.exists()
        ), f"Found nested zarr in folder conversion: {nested_zarr_folder}"

        # Clean up
        shutil.rmtree(zarr_path_folder)

        images = np.random.random((2, 3, 32, 32)).astype(np.float32)

        zarr_path_memory = convert(
            images=images,
            output_dir=temp_dir,
            overwrite=True,
        )

        assert zarr_path_memory.exists()
        assert zarr_path_memory.name == "images.zarr"
        assert zarr_path_memory.parent == temp_dir

        # Check that there's no nested zarr structure
        nested_zarr_memory = zarr_path_memory / "images.zarr"
        assert (
            not nested_zarr_memory.exists()
        ), f"Found nested zarr in memory conversion: {nested_zarr_memory}"
