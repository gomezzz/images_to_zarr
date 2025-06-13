"""Test performance aspects of image conversion."""

import time
import os
import psutil
import numpy as np
import pandas as pd
import zarr
from PIL import Image

from images_to_zarr.convert import convert


class TestPerformance:
    """Basic performance tests."""

    def test_conversion_speed(self, temp_dir):
        """Test conversion speed with a larger dataset."""
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

        # Basic performance check (should process at least 5 images per second)
        images_per_second = num_images / conversion_time
        assert images_per_second > 5, f"Too slow: {images_per_second:.2f} images/sec"

    def test_memory_usage(self, temp_dir, sample_images, sample_metadata):
        """Test that memory usage stays reasonable."""
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
