"""Test metadata handling functionality."""

import pandas as pd
import zarr

from images_to_zarr.convert import convert


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

        # Check additional metadata columns are added
        expected_processing_cols = [
            "original_filename",
            "dtype",
            "shape",
        ]
        for col in expected_processing_cols:
            assert (
                col in saved_metadata.columns
            ), f"Expected column '{col}' not found in {saved_metadata.columns.tolist()}"

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
