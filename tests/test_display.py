"""Test display and visualization functionality."""

import pytest
import numpy as np
import pandas as pd
from astropy.io import fits

from images_to_zarr.convert import convert


class TestDisplay:
    """Test the display_sample_images function."""

    def test_display_sample_images(self, temp_dir, sample_images):
        """Test the display_sample_images function."""
        from images_to_zarr.display_sample_images import display_sample_images

        images_dir, files = sample_images
        output_dir = temp_dir / "output"

        # Create a zarr store
        zarr_path = convert(
            folders=[images_dir],
            output_dir=output_dir,
            overwrite=True,
        )

        # Test basic display (should not raise errors)
        try:
            # Test with default parameters (save to file to avoid blocking during tests)
            save_path1 = temp_dir / "test_display_default.png"
            display_sample_images(zarr_path, num_samples=2, figsize=(8, 6), save_path=save_path1)

            # Test with saving to file
            save_path2 = temp_dir / "test_display.png"
            display_sample_images(zarr_path, num_samples=1, save_path=save_path2)

            # Test with all images if fewer than num_samples (save to file)
            save_path3 = temp_dir / "test_display_many.png"
            display_sample_images(
                zarr_path, num_samples=10, save_path=save_path3
            )  # More than available

        except ImportError:
            # matplotlib not available - this is expected in some test environments
            pytest.skip("matplotlib not available for display testing")
        except Exception as e:
            # The function might fail in a headless environment, but shouldn't crash
            # due to missing display. We mainly want to test the data loading logic.
            if "display" not in str(e).lower() and "DISPLAY" not in str(e):
                raise e

    def test_display_with_different_dtypes(self, temp_dir):
        """Test display_sample_images with different data types and ranges."""
        from images_to_zarr.display_sample_images import display_sample_images

        # Create images with different dtypes
        images_dir = temp_dir / "dtype_test"
        images_dir.mkdir()

        # Create float32 FITS image with values in range [0, 1]
        data_float = np.random.random((64, 64)).astype(np.float32)
        fits_path = images_dir / "float_image.fits"
        hdu = fits.PrimaryHDU(data_float)
        hdu.writeto(fits_path, overwrite=True)

        # Create uint16 image
        data_uint16 = (np.random.random((64, 64)) * 65535).astype(np.uint16)
        fits_path_16 = images_dir / "uint16_image.fits"
        hdu16 = fits.PrimaryHDU(data_uint16)
        hdu16.writeto(fits_path_16, overwrite=True)

        output_dir = temp_dir / "output"
        zarr_path = convert(
            folders=[images_dir],
            output_dir=output_dir,
            overwrite=True,
        )

        # Test display with auto-normalization
        try:
            save_path = temp_dir / "dtype_display.png"
            display_sample_images(zarr_path, num_samples=2, save_path=save_path)
        except ImportError:
            pytest.skip("matplotlib not available for display testing")
        except Exception as e:
            # The function might fail in a headless environment
            if "display" not in str(e).lower() and "DISPLAY" not in str(e):
                raise e

    def test_path_handling(self, temp_dir, sample_images):
        """Test correct path handling for output directories."""
        images_dir, files = sample_images

        # Test 1: Output path ending with .zarr should be used directly
        zarr_output = temp_dir / "custom_name.zarr"
        zarr_path = convert(
            folders=[images_dir],
            output_dir=zarr_output,
            overwrite=True,
        )

        assert zarr_path == zarr_output
        assert zarr_path.exists()

        # Test 2: Output path not ending with .zarr should create .zarr inside
        regular_output = temp_dir / "regular_dir"
        zarr_path = convert(
            folders=[images_dir],
            output_dir=regular_output,
            overwrite=True,
        )

        assert zarr_path.parent == regular_output
        assert zarr_path.name == "images.zarr"
        assert zarr_path.exists()

        # Test 3: With metadata file, should use metadata filename
        metadata_data = [{"filename": f.name} for f in files]
        metadata_df = pd.DataFrame(metadata_data)
        metadata_path = temp_dir / "my_dataset.csv"
        metadata_df.to_csv(metadata_path, index=False)

        zarr_path = convert(
            folders=[images_dir],
            metadata=metadata_path,
            output_dir=regular_output / "with_metadata",
            overwrite=True,
        )

        assert zarr_path.name == "my_dataset.zarr"
