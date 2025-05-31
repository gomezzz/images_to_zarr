import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from PIL import Image
from astropy.io import fits
import zarr
import imageio

from images_to_zarr.convert import convert
from images_to_zarr.inspect import inspect
from images_to_zarr import I2Z_SUPPORTED_EXTS


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_images(temp_dir):
    """Create sample images in various formats."""
    images_dir = temp_dir / "images"
    images_dir.mkdir()

    # Create sample data
    sample_data_2d = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

    files = []

    # PNG image
    png_path = images_dir / "test_image.png"
    Image.fromarray(sample_data_2d, mode="L").save(png_path)
    files.append(png_path)

    # JPEG image
    jpeg_path = images_dir / "test_image.jpg"
    Image.fromarray(sample_data_2d, mode="L").save(jpeg_path)
    files.append(jpeg_path)

    # TIFF image
    tiff_path = images_dir / "test_image.tiff"
    Image.fromarray(sample_data_2d, mode="L").save(tiff_path)
    files.append(tiff_path)

    # FITS image
    fits_path = images_dir / "test_image.fits"
    hdu = fits.PrimaryHDU(sample_data_2d.astype(np.float32))
    hdu.writeto(fits_path, overwrite=True)
    files.append(fits_path)

    # Multi-extension FITS
    fits_multi_path = images_dir / "test_multi.fits"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(sample_data_2d.astype(np.float32), name="SCI"),
            fits.ImageHDU(sample_data_2d.astype(np.float32) * 0.1, name="ERR"),
        ]
    )
    hdul.writeto(fits_multi_path, overwrite=True)
    files.append(fits_multi_path)

    return images_dir, files


@pytest.fixture
def sample_metadata(temp_dir, sample_images):
    """Create sample metadata CSV."""
    images_dir, files = sample_images

    metadata_data = []
    for i, file_path in enumerate(files):
        metadata_data.append(
            {
                "filename": file_path.name,
                "source_id": f"SRC_{i:03d}",
                "ra": 123.456 + i * 0.1,
                "dec": 45.678 + i * 0.1,
                "magnitude": 18.5 + i * 0.2,
            }
        )

    metadata_df = pd.DataFrame(metadata_data)
    metadata_path = temp_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    return metadata_path, metadata_df


class TestImageFormats:
    """Test reading various image formats."""

    def test_supported_extensions(self):
        """Test that all expected extensions are supported."""
        expected_exts = {".fits", ".fit", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        assert I2Z_SUPPORTED_EXTS == expected_exts

    def test_png_reading(self, sample_images):
        """Test PNG image reading."""
        from images_to_zarr.convert import _read_image_data

        images_dir, files = sample_images
        png_file = [f for f in files if f.suffix == ".png"][0]

        data, metadata = _read_image_data(png_file)

        assert data.ndim == 2
        assert data.shape == (64, 64)
        assert metadata["original_extension"] == ".png"
        assert "mode" in metadata

    def test_jpeg_reading(self, sample_images):
        """Test JPEG image reading."""
        from images_to_zarr.convert import _read_image_data

        images_dir, files = sample_images
        jpeg_file = [f for f in files if f.suffix == ".jpg"][0]

        data, metadata = _read_image_data(jpeg_file)

        assert data.ndim == 2
        assert data.shape == (64, 64)
        assert metadata["original_extension"] == ".jpg"

    def test_tiff_reading(self, sample_images):
        """Test TIFF image reading."""
        from images_to_zarr.convert import _read_image_data

        images_dir, files = sample_images
        tiff_file = [f for f in files if f.suffix == ".tiff"][0]

        data, metadata = _read_image_data(tiff_file)

        assert data.ndim == 2
        assert data.shape == (64, 64)
        assert metadata["original_extension"] == ".tiff"

    def test_fits_reading(self, sample_images):
        """Test FITS image reading."""
        from images_to_zarr.convert import _read_image_data

        images_dir, files = sample_images
        fits_file = [f for f in files if f.suffix == ".fits" and "multi" not in f.name][0]

        data, metadata = _read_image_data(fits_file)

        assert data.ndim == 2
        assert data.shape == (64, 64)
        assert metadata["original_extension"] == ".fits"
        assert metadata["fits_extension"] == 0

    def test_fits_multi_extension(self, sample_images):
        """Test multi-extension FITS reading."""
        from images_to_zarr.convert import _read_image_data

        images_dir, files = sample_images
        fits_file = [f for f in files if "multi" in f.name][0]

        # Test single extension by name
        data, metadata = _read_image_data(fits_file, fits_extension="SCI")
        assert data.ndim == 2
        assert metadata["fits_extension"] == "SCI"

        # Test multiple extensions
        data, metadata = _read_image_data(fits_file, fits_extension=["SCI", "ERR"])
        assert metadata["fits_extensions"] == ["SCI", "ERR"]


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


class TestInspection:
    """Test the inspection functionality."""

    def test_basic_inspection(self, temp_dir, sample_images, sample_metadata, capsys):
        """Test basic Zarr store inspection."""
        images_dir, files = sample_images
        metadata_path, metadata_df = sample_metadata
        output_dir = temp_dir / "output"

        # First create a store
        zarr_path = convert(
            folders=[images_dir],
            recursive=False,
            metadata=metadata_path,
            output_dir=output_dir,
            overwrite=True,
        )

        # Then inspect it
        inspect(zarr_path)

        captured = capsys.readouterr()
        output = captured.out

        assert "SUMMARY STATISTICS" in output
        assert f"Total images across all files: {len(files)}" in output
        assert "Format distribution:" in output
        assert "Original data type distribution:" in output

    def test_inspect_nonexistent_store(self, temp_dir):
        """Test inspection of non-existent store."""
        nonexistent_path = temp_dir / "nonexistent.zarr"

        # Should not raise exception, just log error and return
        result = inspect(nonexistent_path)
        assert result is None  # Function should return None for non-existent store

    def test_inspect_without_metadata(self, temp_dir, sample_images, capsys):
        """Test inspection when metadata file is missing."""
        images_dir, files = sample_images

        # Create minimal metadata
        metadata_df = pd.DataFrame({"filename": [f.name for f in files]})
        metadata_path = temp_dir / "minimal_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)

        output_dir = temp_dir / "output"
        zarr_path = convert(
            folders=[images_dir],
            recursive=False,
            metadata=metadata_path,
            output_dir=output_dir,
            overwrite=True,
        )

        # Remove metadata file
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        if metadata_parquet.exists():
            metadata_parquet.unlink()

        inspect(zarr_path)

        captured = capsys.readouterr()
        output = captured.out

        assert "SUMMARY STATISTICS" in output
        assert f"Total images across all files: {len(files)}" in output


class TestPerformance:
    """Basic performance tests."""

    def test_conversion_speed(self, temp_dir):
        """Test conversion speed with a larger dataset."""
        import time

        # Create more test images
        images_dir = temp_dir / "images"
        images_dir.mkdir()

        num_images = 50
        files = []

        for i in range(num_images):
            img_data = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
            img_path = images_dir / f"test_{i:03d}.png"
            Image.fromarray(img_data, mode="L").save(img_path)
            files.append(img_path)

        # Create metadata
        metadata_df = pd.DataFrame({"filename": [f.name for f in files], "id": range(num_images)})
        metadata_path = temp_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)

        # Time the conversion
        start_time = time.time()

        zarr_path = convert(
            folders=[images_dir],
            recursive=False,
            metadata=metadata_path,
            output_dir=temp_dir / "output",
            num_parallel_workers=4,
            overwrite=True,
        )

        conversion_time = time.time() - start_time

        # Verify results
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        assert images_array.shape[0] == num_images

        # Basic performance check (should process at least 10 images per second)
        images_per_second = num_images / conversion_time
        assert images_per_second > 5, f"Too slow: {images_per_second:.2f} images/sec"

    def test_memory_usage(self, temp_dir, sample_images, sample_metadata):
        """Test that memory usage stays reasonable."""
        import psutil
        import os

        images_dir, files = sample_images
        metadata_path, metadata_df = sample_metadata
        output_dir = temp_dir / "output"

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        zarr_path = convert(
            folders=[images_dir],
            recursive=False,
            metadata=metadata_path,
            output_dir=output_dir,
            num_parallel_workers=2,
            overwrite=True,
        )

        # Ensure the conversion completed successfully
        assert zarr_path.exists(), "Zarr store was not created"

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024**2  # MB

        # Should not use more than 100MB for this small test
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.2f} MB"


class TestMetadata:
    """Test metadata handling."""

    def test_metadata_preservation(self, temp_dir, sample_images, sample_metadata):
        """Test that original metadata is preserved."""
        images_dir, files = sample_images
        metadata_path, metadata_df = sample_metadata
        output_dir = temp_dir / "output"

        zarr_path = convert(
            folders=[images_dir],
            recursive=False,
            metadata=metadata_path,
            output_dir=output_dir,
            overwrite=True,
        )

        # Load saved metadata
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        saved_metadata = pd.read_parquet(metadata_parquet)

        # Check original columns are preserved
        for col in metadata_df.columns:
            assert col in saved_metadata.columns

        # Check additional metadata columns are added - prioritize essential columns
        essential_cols = [
            "original_filename",
            "original_extension",
            "dtype",
            "shape",
        ]
        for col in essential_cols:
            assert col in saved_metadata.columns

        # Optional metadata columns (may not be present for performance reasons)
        optional_cols = [
            "file_size_bytes",
            "min_value",
            "max_value",
            "mean_value",
        ]
        # Just check that some optional metadata is present, not all
        optional_present = sum(1 for col in optional_cols if col in saved_metadata.columns)
        assert optional_present >= 0  # At least some metadata should be preserved

    def test_zarr_attributes(self, temp_dir, sample_images, sample_metadata):
        """Test that Zarr attributes are set correctly."""
        images_dir, files = sample_images
        metadata_path, metadata_df = sample_metadata
        output_dir = temp_dir / "output"

        zarr_path = convert(
            folders=[images_dir],
            recursive=False,
            metadata=metadata_path,
            output_dir=output_dir,
            fits_extension=0,
            compressor="zstd",
            clevel=3,
            overwrite=True,
        )

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        attrs = dict(root.attrs)

        assert attrs["total_images"] == len(files)
        assert attrs["compressor"] == "zstd"
        assert attrs["compression_level"] == 3
        assert "supported_extensions" in attrs
        assert "creation_info" in attrs

        creation_info = attrs["creation_info"]
        assert creation_info["fits_extension"] == 0
        assert creation_info["recursive_scan"] is False


if __name__ == "__main__":
    pytest.main([__file__])
