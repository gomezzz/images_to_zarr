"""Test resizing, chunk handling, and folder normalization features."""

import pytest
import numpy as np
import zarr
from PIL import Image

from images_to_zarr.convert import convert


class TestResizingFeatures:
    """Test resizing functionality."""

    def test_resize_functionality(self, temp_dir):
        """Test that resize parameter works correctly."""
        # Create images with different sizes (only 2 to ensure both are sampled)
        images_dir = temp_dir / "mixed_sizes"
        images_dir.mkdir()

        # Create just 2 images with different dimensions to ensure both are analyzed
        sizes = [(32, 48), (64, 64)]
        files = []

        for i, (h, w) in enumerate(sizes):
            data = np.random.randint(0, 255, (h, w), dtype=np.uint8)
            img_path = images_dir / f"image_{i}.png"
            Image.fromarray(data, mode="L").save(img_path)
            files.append(img_path)

        output_dir = temp_dir / "output"

        # Test 1: Without resize, should fail for mismatched dimensions
        with pytest.raises(ValueError, match="All images must have the same dimensions"):
            convert(
                folders=[images_dir],
                output_dir=output_dir,
                overwrite=True,
            )

        # Test 2: With resize, should work and resize all images to target size
        target_size = (50, 60)  # (height, width)
        zarr_path = convert(
            folders=[images_dir],
            output_dir=output_dir,
            resize=target_size,
            overwrite=True,
        )

        # Verify the zarr array has the correct dimensions
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        assert images_array.shape == (len(files), target_size[0], target_size[1])
        assert root.attrs["creation_info"]["resize"] == list(target_size)

    def test_interpolation_order(self, temp_dir):
        """Test different interpolation orders for resizing."""
        # Create a small test image
        images_dir = temp_dir / "test_interp"
        images_dir.mkdir()

        # Create a simple image with clear patterns
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 255  # White square in center
        img_path = images_dir / "test.png"
        Image.fromarray(data, mode="L").save(img_path)

        output_dir = temp_dir / "output"

        # Test different interpolation orders
        for order in [0, 1, 3]:  # Nearest, linear, cubic
            zarr_path = convert(
                folders=[images_dir],
                output_dir=output_dir / f"order_{order}",
                resize=(40, 40),  # Double the size
                interpolation_order=order,
                overwrite=True,
            )

            store = zarr.storage.LocalStore(zarr_path)
            root = zarr.open_group(store=store, mode="r")
            assert root.attrs["creation_info"]["interpolation_order"] == order

        # Test invalid interpolation order
        with pytest.raises(ValueError, match="interpolation_order must be between 0 and 5"):
            convert(
                folders=[images_dir],
                output_dir=output_dir / "invalid",
                resize=(40, 40),
                interpolation_order=10,  # Invalid
                overwrite=True,
            )


class TestChunkShapeHandling:
    """Test chunk_shape parameter handling."""

    def test_chunk_shape_handling(self, temp_dir, sample_images):
        """Test that chunk_shape parameter is handled correctly."""
        images_dir, files = sample_images
        output_dir = temp_dir / "output"

        # Test 1: User-specified 3D chunk shape (should work for 3D arrays)
        user_chunk_3d = (2, 32, 32)
        zarr_path = convert(
            folders=[images_dir],
            output_dir=output_dir / "chunk_3d",
            chunk_shape=user_chunk_3d,
            overwrite=True,
        )

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        # Should respect user input (clamped to array size)
        expected_chunks = tuple(min(c, s) for c, s in zip(user_chunk_3d, images_array.shape))
        assert images_array.chunks == expected_chunks

        # Test 2: User-specified chunk shape that's too large (should be clamped)
        large_chunk = (100, 200, 200)  # Larger than image dimensions
        zarr_path = convert(
            folders=[images_dir],
            output_dir=output_dir / "chunk_large",
            chunk_shape=large_chunk,
            overwrite=True,
        )

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        # Should be clamped to actual array size
        expected_chunks = tuple(min(c, s) for c, s in zip(large_chunk, images_array.shape))
        assert images_array.chunks == expected_chunks

    def test_chunk_shape_with_channels(self, temp_dir):
        """Test chunk_shape handling with multi-channel images."""
        # Create RGB images
        images_dir = temp_dir / "rgb_images"
        images_dir.mkdir()

        # Create 3-channel RGB images
        for i in range(3):
            data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img_path = images_dir / f"rgb_{i}.png"
            Image.fromarray(data, mode="RGB").save(img_path)

        output_dir = temp_dir / "output"

        # Test 3D chunk shape with 4D array (should expand to include channels)
        user_chunk_3d = (1, 32, 32)
        zarr_path = convert(
            folders=[images_dir],
            output_dir=output_dir,
            chunk_shape=user_chunk_3d,
            overwrite=True,
        )

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        # Should be 4D array with channels
        assert len(images_array.shape) == 4  # (N, C, H, W)
        assert images_array.shape[1] == 3  # 3 channels
        # Chunk shape should be expanded to include channels
        expected_chunks = (1, 3, 32, 32)  # Full channels dimension
        assert images_array.chunks == expected_chunks


class TestFoldersInputNormalization:
    """Test that single string folders input is converted to list."""

    def test_single_string_folder(self, temp_dir, sample_images):
        """Test that a single string folder is converted to a list."""
        images_dir, files = sample_images

        # Test with single string
        zarr_path = convert(
            output_dir=temp_dir,
            folders=str(images_dir),  # Single string, not list
            overwrite=True,
        )

        # Should work and create a zarr store
        assert zarr_path.exists()
        assert zarr_path.is_dir()

    def test_single_path_folder(self, temp_dir, sample_images):
        """Test that a single Path folder is converted to a list."""
        images_dir, files = sample_images

        # Test with single Path object
        zarr_path = convert(
            output_dir=temp_dir,
            folders=images_dir,  # Single Path, not list
            overwrite=True,
        )

        # Should work and create a zarr store
        assert zarr_path.exists()
        assert zarr_path.is_dir()

    def test_list_of_folders(self, temp_dir, sample_images):
        """Test that a list of folders works correctly."""
        images_dir, files = sample_images

        # Create another folder with one image
        images_dir2 = temp_dir / "images2"
        images_dir2.mkdir()
        sample_data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        Image.fromarray(sample_data, mode="L").save(images_dir2 / "extra.png")

        # Test with list of folders
        zarr_path = convert(
            output_dir=temp_dir,
            folders=[images_dir, images_dir2],  # List of folders
            overwrite=True,
        )

        # Should work and create a zarr store with images from both folders
        assert zarr_path.exists()
        assert zarr_path.is_dir()

        # Check that we have images from both folders
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        assert images_array.shape[0] == len(files) + 1  # Original files + 1 extra
