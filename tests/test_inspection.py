"""Test inspection functionality."""

import pandas as pd

from images_to_zarr.convert import convert
from images_to_zarr.inspect import inspect


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
        # Format and data type distribution may or may not be present depending on metadata
        # Just check that basic summary information is there
        assert "Data type:" in output

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
