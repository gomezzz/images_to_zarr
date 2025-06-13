import sys
from pathlib import Path
from images_to_zarr import configure_logging
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from PIL import Image
from astropy.io import fits
import zarr
import imageio

from images_to_zarr import I2Z_SUPPORTED_EXTS

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
configure_logging(enable=True, level="WARNING")  # Reduce noise in tests


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
