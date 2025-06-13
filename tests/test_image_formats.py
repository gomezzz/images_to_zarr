"""Test image format reading functionality."""

from images_to_zarr import I2Z_SUPPORTED_EXTS


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
