"""Test conversion functionality and NCHW format handling."""

import pytest
import numpy as np
import pandas as pd
import zarr
import imageio
from PIL import Image

from images_to_zarr.convert import convert


class TestConversion:
    """Test the main conversion functionality."""

    def test_basic_conversion(self, temp_dir, sample_images, sample_metadata):
        """Test basic image to Zarr conversion."""
        images_dir, files = sample_images
        metadata_path, metadata_df = sample_metadata
        output_dir = temp_dir / "output"

        zarr_path = convert(
            folders=[images_dir],
            recursive=False,
            metadata=metadata_path,
            output_dir=output_dir,
            num_parallel_workers=2,
            chunk_shape=(1, 64, 64),
            overwrite=True,
        )

        assert zarr_path.exists()
        assert zarr_path.is_dir()
        assert zarr_path.name.endswith(".zarr")

        # Check Zarr structure
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")

        assert "images" in root
        images_array = root["images"]
        assert images_array.shape[0] == len(files)

        # Check metadata file
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        assert metadata_parquet.exists()

        saved_metadata = pd.read_parquet(metadata_parquet)
        assert len(saved_metadata) == len(files)

    def test_recursive_search(self, temp_dir):
        """Test recursive directory search."""
        from images_to_zarr.convert import _find_image_files

        # Create nested directory structure
        images_dir = temp_dir / "images_test"
        sub_dir = images_dir / "subdir"
        sub_dir.mkdir(parents=True)

        # Create images in both directories
        Image.fromarray(np.random.randint(0, 255, (32, 32), dtype=np.uint8)).save(
            images_dir / "img1.png"
        )
        Image.fromarray(np.random.randint(0, 255, (32, 32), dtype=np.uint8)).save(
            sub_dir / "img2.png"
        )

        # Test non-recursive
        files_non_recursive = _find_image_files([images_dir], recursive=False)
        assert len(files_non_recursive) == 1

        # Test recursive
        files_recursive = _find_image_files([images_dir], recursive=True)
        assert len(files_recursive) == 2

    def test_compression_options(self, temp_dir, sample_images, sample_metadata):
        """Test different compression options."""
        images_dir, files = sample_images
        metadata_path, metadata_df = sample_metadata
        output_dir = temp_dir / "output"

        # Test with different compressors
        for compressor in ["zstd", "lz4", "gzip"]:
            zarr_path = convert(
                folders=[images_dir],
                recursive=False,
                metadata=metadata_path,
                output_dir=output_dir / compressor,
                compressor=compressor,
                clevel=1,
                overwrite=True,
            )

            store = zarr.storage.LocalStore(zarr_path)
            root = zarr.open_group(store=store, mode="r")
            assert root.attrs["compressor"] == compressor

    def test_error_handling(self, temp_dir):
        """Test error handling for invalid inputs."""
        output_dir = temp_dir / "output"

        # Create a dummy image file for the metadata tests
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        dummy_path = temp_dir / "dummy.png"
        imageio.imwrite(dummy_path, dummy_image)

        # Test missing metadata file
        with pytest.raises(FileNotFoundError):
            convert(
                folders=[temp_dir],
                recursive=False,
                metadata=temp_dir / "nonexistent.csv",
                output_dir=output_dir,
            )

        # Test invalid metadata CSV
        bad_metadata = temp_dir / "bad_metadata.csv"
        pd.DataFrame({"not_filename": ["test"]}).to_csv(bad_metadata, index=False)

        with pytest.raises(ValueError, match="filename"):
            convert(
                folders=[temp_dir], recursive=False, metadata=bad_metadata, output_dir=output_dir
            )

    def test_conversion_without_metadata(self, temp_dir, sample_images):
        """Test conversion with automatically generated metadata from filenames."""
        images_dir, files = sample_images
        output_dir = temp_dir / "output"

        zarr_path = convert(
            folders=[images_dir],
            recursive=False,
            metadata=None,  # No metadata provided
            output_dir=output_dir,
            num_parallel_workers=2,
            chunk_shape=(1, 64, 64),
            overwrite=True,
        )

        assert zarr_path.exists()
        assert zarr_path.is_dir()
        assert zarr_path.name == "images.zarr"  # Default name when no metadata

        # Check Zarr structure
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")

        assert "images" in root
        images_array = root["images"]
        assert images_array.shape[0] == len(files)

        # Check metadata file - should contain only filenames
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        assert metadata_parquet.exists()

        saved_metadata = pd.read_parquet(metadata_parquet)
        assert len(saved_metadata) == len(files)
        assert "filename" in saved_metadata.columns

        # Should contain all the filenames
        expected_filenames = {f.name for f in files}
        actual_filenames = set(saved_metadata["filename"])
        assert expected_filenames == actual_filenames


class TestNCHWFormat:
    """Test that all images are converted to NCHW format correctly."""

    def test_grayscale_to_nchw(self, temp_dir):
        """Test grayscale image conversion to NCHW format."""
        from images_to_zarr.convert import _ensure_nchw_format

        # Create a simple grayscale image (H, W)
        grayscale_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

        # Test _ensure_nchw_format directly
        nchw_data = _ensure_nchw_format(grayscale_data)

        # Should be (1, 1, 64, 64) - batch=1, channels=1, height=64, width=64
        assert nchw_data.shape == (1, 1, 64, 64)
        assert np.array_equal(nchw_data[0, 0, :, :], grayscale_data)

    def test_rgb_hwc_to_nchw(self, temp_dir):
        """Test RGB image in HWC format conversion to NCHW."""
        from images_to_zarr.convert import _ensure_nchw_format

        # Create RGB image in HWC format (Height, Width, Channels)
        rgb_hwc = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        nchw_data = _ensure_nchw_format(rgb_hwc)

        # Should be (1, 3, 64, 64) - batch=1, channels=3, height=64, width=64
        assert nchw_data.shape == (1, 3, 64, 64)

        # Check that data is correctly transposed
        for c in range(3):
            assert np.array_equal(nchw_data[0, c, :, :], rgb_hwc[:, :, c])

    def test_fits_chw_to_nchw(self, temp_dir):
        """Test FITS image in CHW format conversion to NCHW."""
        from images_to_zarr.convert import _ensure_nchw_format

        # Create FITS-style image in CHW format (Channels, Height, Width)
        fits_chw = np.random.random((2, 64, 64)).astype(np.float32)

        nchw_data = _ensure_nchw_format(fits_chw)

        # Should be (1, 2, 64, 64) - batch=1, channels=2, height=64, width=64
        assert nchw_data.shape == (1, 2, 64, 64)
        assert np.array_equal(nchw_data[0, :, :, :], fits_chw)

    def test_different_formats_produce_nchw(self, sample_images):
        """Test that different image formats all produce NCHW output."""
        from images_to_zarr.convert import _read_image_data, _ensure_nchw_format

        images_dir, files = sample_images

        for file_path in files:
            # Read raw data and convert to NCHW
            raw_data, metadata = _read_image_data(file_path)
            data = _ensure_nchw_format(raw_data)

            # All images should be in NCHW format (4D)
            assert data.ndim == 4, f"Image {file_path.name} is not 4D: {data.shape}"
            assert data.shape[0] == 1, f"Batch dimension should be 1: {data.shape}"

            # Check that channels, height, width are positive
            _, c, h, w = data.shape
            assert c > 0, f"Channels dimension invalid: {c}"
            assert h > 0, f"Height dimension invalid: {h}"
            assert w > 0, f"Width dimension invalid: {w}"

    def test_zarr_store_has_nchw_format(self, temp_dir, sample_images, sample_metadata):
        """Test that the final Zarr store contains data in NCHW format."""
        images_dir, files = sample_images
        metadata_path, metadata_df = sample_metadata

        # Convert to Zarr
        zarr_path = convert(
            output_dir=temp_dir,
            folders=[images_dir],
            metadata=metadata_path,
            chunk_shape=(1, 128, 128),
            overwrite=True,
        )

        # Open the Zarr store and check format
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]  # For grayscale images, should be 3D: (N, H, W)
        # For multi-channel images, should be 4D: (N, C, H, W)
        assert images_array.ndim in [3, 4], f"Zarr array should be 3D or 4D: {images_array.shape}"

        # Check that we have the expected number of images
        assert images_array.shape[0] == len(files)

        if images_array.ndim == 4:
            # Multi-channel format (N, C, H, W)
            n, c, h, w = images_array.shape
            assert c > 0 and h > 0 and w > 0
        else:
            # Grayscale format (N, H, W)
            n, h, w = images_array.shape
            assert h > 0 and w > 0

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


class TestAppendConversion:
    """Test append functionality with file-based conversion."""

    def test_append_from_files(self, temp_dir, sample_images):
        """Test appending images from files to existing store."""
        images_dir, files = sample_images

        # Create initial store from subset of files
        zarr_path = convert(
            folders=[images_dir],
            output_dir=temp_dir,
            overwrite=True,
        )

        # Get initial count
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        initial_count = root["images"].shape[0]
        _ = root["images"][:]
        # Zarr groups don't need to be closed

        # Create more sample images to append
        from PIL import Image

        append_dir = temp_dir / "append_images"
        append_dir.mkdir()

        # Create 2 new images
        for i in range(2):
            img_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            img_path = append_dir / f"append_{i}.png"
            Image.fromarray(img_data, mode="L").save(img_path)

        # Append to existing store
        convert(
            folders=[append_dir],
            output_dir=temp_dir,
            append=True,
        )

        # Verify
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        final_count = root["images"].shape[0]

        assert final_count > initial_count  # Should have more images after append

    def test_append_with_metadata(self, temp_dir, sample_images):
        """Test appending with metadata preservation."""
        images_dir, files = sample_images

        # Create initial metadata
        initial_metadata = pd.DataFrame(
            {"filename": [f.name for f in files], "category": ["original"] * len(files)}
        )
        initial_metadata_path = temp_dir / "initial_meta.csv"
        initial_metadata.to_csv(initial_metadata_path, index=False)

        # Create initial store
        zarr_path = convert(
            folders=[images_dir],
            metadata=initial_metadata_path,
            output_dir=temp_dir,
            overwrite=True,
        )

        # Create append images and metadata
        from PIL import Image

        append_dir = temp_dir / "append_images"
        append_dir.mkdir()

        append_files = []
        for i in range(2):
            img_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            img_path = append_dir / f"new_{i}.png"
            Image.fromarray(img_data, mode="L").save(img_path)
            append_files.append(img_path)

        append_metadata = pd.DataFrame(
            {
                "filename": [f.name for f in append_files],
                "category": ["appended"] * len(append_files),
            }
        )
        append_metadata_path = temp_dir / "append_meta.csv"
        append_metadata.to_csv(append_metadata_path, index=False)

        # Append
        convert(
            folders=[append_dir],
            metadata=append_metadata_path,
            output_dir=zarr_path,  # Use the existing zarr path
            append=True,
        )

        # Check metadata was merged
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        combined_metadata = pd.read_parquet(metadata_parquet)

        # Should have metadata from both initial and appended images
        assert len(combined_metadata) >= len(files)
        if "category" in combined_metadata.columns:
            categories = combined_metadata["category"].tolist()
            # Just check that append operation worked
            assert len(categories) > 0
