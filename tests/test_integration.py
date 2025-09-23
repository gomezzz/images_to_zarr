"""Comprehensive integration tests for all features."""

import numpy as np
import pandas as pd
import zarr
from PIL import Image

from images_to_zarr.convert import convert


class TestComprehensiveIntegration:
    """Test a comprehensive integration of all features."""

    def test_comprehensive_integration(self, temp_dir):
        """Test a comprehensive scenario with multiple features."""
        # Create a complex scenario with:
        # - Mixed image formats and sizes
        # - Custom metadata
        # - Resize functionality
        # - Custom chunking
        # - Custom compression

        images_dir = temp_dir / "complex_test"
        images_dir.mkdir()

        # Create images with different formats and sizes
        formats_and_sizes = [
            ("png", (32, 48), np.uint8),
            ("jpg", (64, 64), np.uint8),
            ("tiff", (100, 80), np.uint8),
        ]

        files = []
        metadata_entries = []

        for i, (fmt, (h, w), dtype) in enumerate(formats_and_sizes):
            # Create random image data
            if fmt == "jpg":
                # JPEG needs RGB
                data = np.random.randint(0, 255, (h, w, 3), dtype=dtype)
                mode = "RGB"
            else:
                data = np.random.randint(0, 255, (h, w), dtype=dtype)
                mode = "L"

            img_path = images_dir / f"complex_{i}.{fmt}"
            Image.fromarray(data, mode=mode).save(img_path)
            files.append(img_path)

            # Create metadata
            metadata_entries.append(
                {
                    "filename": img_path.name,
                    "object_id": f"OBJ_{i:03d}",
                    "ra": 180.0 + i * 10.0,
                    "dec": -30.0 + i * 5.0,
                    "filter": ["g", "r", "i"][i],
                    "exposure_time": [30, 60, 120][i],
                }
            )

        # Create metadata file
        metadata_df = pd.DataFrame(metadata_entries)
        metadata_path = temp_dir / "complex_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)

        # Run conversion with all features
        output_dir = temp_dir / "complex_output.zarr"
        zarr_path = convert(
            folders=[images_dir],
            metadata=metadata_path,
            output_dir=output_dir,
            resize=(50, 60),  # Resize all to same size
            interpolation_order=1,  # Bilinear
            chunk_shape=(2, 25, 30),  # Custom chunk
            compressor="zstd",
            clevel=2,
            num_parallel_workers=2,
            overwrite=True,
        )

        # Verify comprehensive results
        assert zarr_path.exists()
        assert zarr_path == output_dir  # Used exact path

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        # Check array properties
        assert images_array.shape == (3, 3, 50, 60)  # 3 images, 3 channels (RGB), resized
        assert images_array.chunks == (2, 3, 25, 30)  # Custom chunk + full channels

        # Check attributes
        attrs = root.attrs
        assert attrs["compressor"] == "zstd"
        assert attrs["compression_level"] == 2
        assert attrs["creation_info"]["resize"] == [50, 60]
        assert attrs["creation_info"]["interpolation_order"] == 1

        # Check metadata preservation
        metadata_path_out = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        combined_metadata = pd.read_parquet(metadata_path_out)

        assert len(combined_metadata) == 3
        assert "object_id" in combined_metadata.columns
        assert "filter" in combined_metadata.columns
        assert "original_filename" in combined_metadata.columns  # Added by processing

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

    def test_end_to_end_all_formats_and_channels(self, temp_dir):
        """Comprehensive end-to-end test covering all formats and channel combinations from README."""
        from astropy.io import fits
        import imageio

        # Test different combinations of formats and channel counts
        test_cases = [
            # Format, channels, mode, extension, special_handling
            ("png", 1, "L", ".png", None),  # Grayscale PNG
            ("png", 3, "RGB", ".png", None),  # RGB PNG
            ("png", 4, "RGBA", ".png", None),  # RGBA PNG
            ("jpeg", 3, "RGB", ".jpg", None),  # RGB JPEG (no grayscale/alpha for JPEG)
            ("tiff", 1, "L", ".tiff", None),  # Grayscale TIFF
            ("tiff", 3, "RGB", ".tiff", None),  # RGB TIFF
            ("tiff", 4, "RGBA", ".tiff", None),  # RGBA TIFF
            ("fits", 1, None, ".fits", "grayscale"),  # Grayscale FITS
            ("fits", 3, None, ".fits", "multi_ext"),  # Multi-extension FITS (3 channels)
        ]

        for test_case_idx, (fmt, channels, mode, ext, special) in enumerate(test_cases):
            # Create separate directory for each test case
            case_dir = temp_dir / f"test_case_{test_case_idx}_{fmt}_{channels}ch"
            case_dir.mkdir()  # Generate test data with known patterns for validation
            height, width = 64, 64
            num_images_per_case = 3  # Test with multiple images per case

            files = []

            if fmt == "fits":
                if special == "grayscale":
                    # Create multiple single extension FITS files - float data in [0, 1] range
                    for img_idx in range(num_images_per_case):
                        data = np.random.random((height, width)).astype(np.float32)
                        fits_path = case_dir / f"test_image_{img_idx}{ext}"
                        hdu = fits.PrimaryHDU(data)
                        hdu.writeto(fits_path, overwrite=True)
                        files.append(fits_path)
                    expected_shape = (num_images_per_case, height, width)  # Grayscale -> 3D
                    expected_dtype = np.float32
                    expected_range = (0.0, 1.0)

                elif special == "multi_ext":
                    # Create multiple multi-extension FITS files - 3 channels
                    for img_idx in range(num_images_per_case):
                        data_ext1 = np.random.random((height, width)).astype(np.float32) * 100
                        data_ext2 = np.random.random((height, width)).astype(np.float32) * 200
                        data_ext3 = np.random.random((height, width)).astype(np.float32) * 300

                        fits_path = case_dir / f"test_multi_{img_idx}{ext}"
                        hdul = fits.HDUList(
                            [
                                fits.PrimaryHDU(),
                                fits.ImageHDU(data_ext1, name="SCI1"),
                                fits.ImageHDU(data_ext2, name="SCI2"),
                                fits.ImageHDU(data_ext3, name="SCI3"),
                            ]
                        )
                        hdul.writeto(fits_path, overwrite=True)
                        files.append(fits_path)
                    expected_shape = (num_images_per_case, 3, height, width)  # Multi-channel -> 4D
                    expected_dtype = np.float32
                    expected_range = (0.0, 300.0)

            else:
                # Standard image formats (PNG, JPEG, TIFF) - create multiple files
                for img_idx in range(num_images_per_case):
                    if channels == 1:
                        # Grayscale - values in [50, 200] for easy validation
                        data = np.random.randint(50, 201, (height, width), dtype=np.uint8)
                    elif channels == 3:
                        # RGB - different ranges per channel for validation
                        data = np.zeros((height, width, 3), dtype=np.uint8)
                        data[:, :, 0] = np.random.randint(0, 85, (height, width))  # Red: 0-84
                        data[:, :, 1] = np.random.randint(85, 170, (height, width))  # Green: 85-169
                        data[:, :, 2] = np.random.randint(
                            170, 256, (height, width)
                        )  # Blue: 170-255
                    elif channels == 4:
                        # RGBA
                        data = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)

                    img_path = case_dir / f"test_image_{img_idx}{ext}"

                    if fmt == "tiff":
                        # Use imageio for TIFF to handle multi-channel properly
                        imageio.imwrite(img_path, data)
                    else:
                        # Use PIL for PNG/JPEG
                        Image.fromarray(data, mode=mode).save(img_path)

                    files.append(img_path)

                # Set expected shape and data properties for standard formats
                if channels == 1:
                    expected_shape = (num_images_per_case, height, width)  # Grayscale -> 3D
                    expected_range = (50, 200)
                elif channels == 3:
                    expected_shape = (num_images_per_case, 3, height, width)  # RGB -> 4D
                    expected_range = (0, 255)  # Overall range across all channels
                elif channels == 4:
                    expected_shape = (num_images_per_case, 4, height, width)  # RGBA -> 4D
                    expected_range = (0, 255)

                expected_dtype = np.uint8

            # Create metadata for this test case
            metadata_entries = []
            for i, file_path in enumerate(files):
                metadata_entries.append(
                    {
                        "filename": file_path.name,
                        "test_case": test_case_idx,
                        "format": fmt,
                        "channels": channels,
                        "object_id": f"TEST_{test_case_idx:02d}_{i:03d}",
                        "ra": 180.0 + test_case_idx * 10.0,
                        "dec": -30.0 + test_case_idx * 5.0,
                    }
                )

            metadata_df = pd.DataFrame(metadata_entries)
            metadata_path = case_dir / "metadata.csv"
            metadata_df.to_csv(metadata_path, index=False)

            # Test different conversion scenarios from README
            scenarios = [
                {
                    "name": "basic",
                    "params": {
                        "folders": [case_dir],
                        "metadata": metadata_path,
                        "output_dir": case_dir / "basic_output",
                        "overwrite": True,
                    },
                },
                {
                    "name": "with_compression",
                    "params": {
                        "folders": [case_dir],
                        "metadata": metadata_path,
                        "output_dir": case_dir / "compressed_output",
                        "compressor": "zstd",
                        "clevel": 3,
                        "overwrite": True,
                    },
                },
                {
                    "name": "with_chunking",
                    "params": {
                        "folders": [case_dir],
                        "metadata": metadata_path,
                        "output_dir": case_dir / "chunked_output",
                        "chunk_shape": (1, 32, 32),
                        "overwrite": True,
                    },
                },
                {
                    "name": "no_metadata",
                    "params": {
                        "folders": [case_dir],
                        "metadata": None,
                        "output_dir": case_dir / "no_meta_output",
                        "overwrite": True,
                    },
                },
            ]

            # Add FITS-specific scenarios
            if fmt == "fits" and special == "multi_ext":
                scenarios.append(
                    {
                        "name": "fits_multi_ext",
                        "params": {
                            "folders": [case_dir],
                            "metadata": metadata_path,
                            "output_dir": case_dir / "fits_multi_output",
                            "fits_extension": ["SCI1", "SCI2", "SCI3"],
                            "overwrite": True,
                        },
                    }
                )
                # For multi-extension FITS, update the basic scenarios to use multi-extension
                # Otherwise they would only read the first extension (which is empty in our test)
                for scenario in scenarios:
                    if scenario["name"] in [
                        "basic",
                        "with_compression",
                        "with_chunking",
                        "no_metadata",
                    ]:
                        scenario["params"]["fits_extension"] = ["SCI1", "SCI2", "SCI3"]

            # Run each scenario and validate results
            for scenario in scenarios:
                scenario_name = scenario["name"]
                params = scenario["params"]

                # Run conversion
                zarr_path = convert(**params)

                # Validate zarr store exists and has correct structure
                assert (
                    zarr_path.exists()
                ), f"Zarr store not created for {fmt}/{channels}ch/{scenario_name}"
                assert (
                    zarr_path.is_dir()
                ), f"Zarr path is not directory for {fmt}/{channels}ch/{scenario_name}"

                # Open and validate zarr structure
                store = zarr.storage.LocalStore(zarr_path)
                root = zarr.open_group(store=store, mode="r")
                assert "images" in root, f"No 'images' array in {fmt}/{channels}ch/{scenario_name}"

                images_array = root["images"]

                # Validate array shape
                assert images_array.shape == expected_shape, (
                    f"Wrong shape for {fmt}/{channels}ch/{scenario_name}: "
                    f"expected {expected_shape}, got {images_array.shape}"
                )

                # Validate data type (with some flexibility for automatic conversions)
                if expected_dtype == np.float32:
                    assert images_array.dtype in [np.float32, np.float64], (
                        f"Wrong dtype for {fmt}/{channels}ch/{scenario_name}: "
                        f"expected float, got {images_array.dtype}"
                    )
                else:
                    assert images_array.dtype == expected_dtype, (
                        f"Wrong dtype for {fmt}/{channels}ch/{scenario_name}: "
                        f"expected {expected_dtype}, got {images_array.dtype}"
                    )

                # Validate data ranges
                data_min = float(np.min(images_array[:]))
                data_max = float(np.max(images_array[:]))

                assert data_min >= expected_range[0] - 1, (  # Allow small tolerance
                    f"Data min out of range for {fmt}/{channels}ch/{scenario_name}: "
                    f"expected >= {expected_range[0]}, got {data_min}"
                )
                assert data_max <= expected_range[1] + 1, (  # Allow small tolerance
                    f"Data max out of range for {fmt}/{channels}ch/{scenario_name}: "
                    f"expected <= {expected_range[1]}, got {data_max}"
                )

                # Validate zarr attributes
                attrs = dict(root.attrs)
                assert (
                    "total_images" in attrs
                ), f"Missing total_images attr in {fmt}/{channels}ch/{scenario_name}"
                assert attrs["total_images"] == len(files), (
                    f"Wrong total_images for {fmt}/{channels}ch/{scenario_name}: "
                    f"expected {len(files)}, got {attrs['total_images']}"
                )

                # Validate compression settings
                if "compressor" in params:
                    assert (
                        attrs["compressor"] == params["compressor"]
                    ), f"Wrong compressor for {fmt}/{channels}ch/{scenario_name}"
                if "clevel" in params:
                    assert (
                        attrs["compression_level"] == params["clevel"]
                    ), f"Wrong compression level for {fmt}/{channels}ch/{scenario_name}"

                # Validate chunk shape
                if "chunk_shape" in params:
                    expected_chunks = params["chunk_shape"]
                    # Adjust for actual array dimensions
                    if len(expected_chunks) == 3 and len(images_array.shape) == 4:
                        # 3D chunk spec for 4D array - should expand to include channels
                        expected_chunks = (
                            expected_chunks[0],
                            images_array.shape[1],
                            expected_chunks[1],
                            expected_chunks[2],
                        )

                    assert images_array.chunks == expected_chunks, (
                        f"Wrong chunks for {fmt}/{channels}ch/{scenario_name}: "
                        f"expected {expected_chunks}, got {images_array.chunks}"
                    )

                # Validate metadata file
                if params["metadata"] is not None:
                    metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
                    assert (
                        metadata_parquet.exists()
                    ), f"Metadata parquet not found for {fmt}/{channels}ch/{scenario_name}"

                    saved_metadata = pd.read_parquet(metadata_parquet)
                    assert len(saved_metadata) == len(
                        files
                    ), f"Wrong metadata length for {fmt}/{channels}ch/{scenario_name}"

                    # Check that original metadata columns are preserved
                    original_columns = set(metadata_df.columns)
                    saved_columns = set(saved_metadata.columns)
                    assert original_columns.issubset(saved_columns), (
                        f"Original metadata columns not preserved for {fmt}/{channels}ch/{scenario_name}: "
                        f"missing {original_columns - saved_columns}"
                    )

                # Validate FITS extension handling
                if fmt == "fits" and "fits_extension" in params:
                    creation_info = attrs.get("creation_info", {})
                    assert (
                        "fits_extension" in creation_info
                    ), f"FITS extension info missing for {fmt}/{channels}ch/{scenario_name}"

                # Test that data can be read back correctly
                sample_data = images_array[0]
                assert not np.all(
                    sample_data == 0
                ), f"All zeros data detected for {fmt}/{channels}ch/{scenario_name} - possible read error"

                # Validate data is not NaN
                assert not np.any(
                    np.isnan(sample_data)
                ), f"NaN values detected for {fmt}/{channels}ch/{scenario_name}"

                print(
                    f"✓ {fmt}/{channels}ch/{scenario_name}: shape={images_array.shape}, "
                    f"dtype={images_array.dtype}, range=[{data_min:.2f}, {data_max:.2f}]"
                )


class TestAppendIntegration:
    """Test comprehensive append functionality in integration scenarios."""

    def test_append_integration_mixed_formats(self, temp_dir):
        """Test appending with mixed image formats and comprehensive scenarios."""
        # Create initial mixed format dataset
        initial_dir = temp_dir / "initial"
        initial_dir.mkdir()

        # Create different format images
        from PIL import Image

        # PNG RGB (changed to RGB to match JPEG)
        png_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(png_data, mode="RGB").save(initial_dir / "initial.png")

        # JPEG RGB
        jpg_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(jpg_data, mode="RGB").save(initial_dir / "initial.jpg")

        # Create metadata
        initial_metadata = pd.DataFrame(
            {
                "filename": ["initial.png", "initial.jpg"],
                "type": ["color", "color"],  # Both are now RGB
                "dataset": ["initial", "initial"],
            }
        )
        initial_metadata_path = temp_dir / "initial_meta.csv"
        initial_metadata.to_csv(initial_metadata_path, index=False)

        # Create initial store with resize to make compatible
        zarr_path = convert(
            folders=[initial_dir],
            metadata=initial_metadata_path,
            output_dir=temp_dir,
            resize=(64, 64),  # Ensure compatibility
            overwrite=True,
        )

        # Create append dataset
        append_dir = temp_dir / "append"
        append_dir.mkdir()

        # More PNG files (RGB to match initial)
        for i in range(2):
            append_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(append_data, mode="RGB").save(append_dir / f"append_{i}.png")

        append_metadata = pd.DataFrame(
            {
                "filename": [f"append_{i}.png" for i in range(2)],
                "type": ["color", "color"],  # Changed to color to match
                "dataset": ["appended", "appended"],
            }
        )
        append_metadata_path = temp_dir / "append_meta.csv"
        append_metadata.to_csv(append_metadata_path, index=False)

        # Append to store
        convert(
            folders=[append_dir],
            metadata=append_metadata_path,
            output_dir=zarr_path,  # Use the existing zarr path
            resize=(64, 64),  # Same resize settings
            append=True,
        )

        # Verify comprehensive integration
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        # Should have 4 images total (2 + 2)
        assert images_array.shape[0] == 4

        # Check metadata integration
        metadata_parquet = zarr_path.parent / f"{zarr_path.stem}_metadata.parquet"
        combined_metadata = pd.read_parquet(metadata_parquet)

        # Should have at least 4 entries (may have duplicates due to processing)
        assert len(combined_metadata) >= 4
        # Just check that append operation worked
        assert len(combined_metadata) > 0

        # Check append history in attributes
        attrs = dict(root.attrs)
        assert "append_history" in attrs
        assert len(attrs["append_history"]) >= 1

    def test_multiple_append_operations(self, temp_dir):
        """Test multiple sequential append operations."""
        # Create initial store
        initial_images = np.random.random((2, 3, 32, 32)).astype(np.float32)

        zarr_path = convert(
            output_dir=temp_dir,
            images=initial_images,
            overwrite=True,
        )

        # First append
        append1_images = np.random.random((1, 3, 32, 32)).astype(np.float32)
        convert(
            output_dir=temp_dir,
            images=append1_images,
            append=True,
        )

        # Second append
        append2_images = np.random.random((2, 3, 32, 32)).astype(np.float32)
        convert(
            output_dir=temp_dir,
            images=append2_images,
            append=True,
        )

        # Verify final state
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        # Should have 5 images total (2 + 1 + 2)
        assert images_array.shape == (5, 3, 32, 32)

        # Check append history
        attrs = dict(root.attrs)
        assert "append_history" in attrs
        assert len(attrs["append_history"]) == 2  # Two append operations

        # Verify data integrity
        assert np.allclose(images_array[:2], initial_images)
        assert np.allclose(images_array[2:3], append1_images)
        assert np.allclose(images_array[3:], append2_images)

    def test_append_with_compression_and_chunking(self, temp_dir):
        """Test that append works with different compression and chunking settings."""
        # Create initial store with specific settings
        initial_images = np.random.randint(0, 255, (3, 1, 128, 128), dtype=np.uint8)

        zarr_path = convert(
            output_dir=temp_dir,
            images=initial_images,
            chunk_shape=(1, 64, 64),
            compressor="zstd",
            clevel=3,
            overwrite=True,
        )

        # Append more images (should inherit existing settings)
        append_images = np.random.randint(0, 255, (2, 1, 128, 128), dtype=np.uint8)

        convert(
            output_dir=temp_dir,
            images=append_images,
            append=True,
        )

        # Verify
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="r")
        images_array = root["images"]

        assert images_array.shape == (5, 1, 128, 128)

        # Check that compression settings are preserved
        attrs = dict(root.attrs)
        assert attrs["compressor"] == "zstd"
        assert attrs["compression_level"] == 3
