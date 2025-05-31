import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from click.testing import CliRunner

from images_to_zarr.CLI import main


@pytest.fixture
def cli_test_setup():
    """Set up test environment for CLI tests."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create test images
        images_dir = temp_dir / "images"
        images_dir.mkdir()

        files = []
        for i in range(3):
            img_data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
            img_path = images_dir / f"test_{i:03d}.png"
            Image.fromarray(img_data, mode="L").save(img_path)
            files.append(img_path)

        # Create metadata
        metadata_df = pd.DataFrame(
            {
                "filename": [f.name for f in files],
                "source_id": [f"SRC_{i}" for i in range(len(files))],
                "magnitude": [18.5 + i * 0.1 for i in range(len(files))],
            }
        )
        metadata_path = temp_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        yield {
            "temp_dir": temp_dir,
            "images_dir": images_dir,
            "metadata_path": metadata_path,
            "output_dir": output_dir,
            "files": files,
        }
    finally:
        shutil.rmtree(temp_dir)


class TestCLI:
    """Test the command line interface."""

    def test_convert_command(self, cli_test_setup):
        """Test the convert CLI command."""
        setup = cli_test_setup
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                "convert",
                str(setup["images_dir"]),
                "--metadata",
                str(setup["metadata_path"]),
                "--out",
                str(setup["output_dir"]),
                "--workers",
                "2",
                "--chunk-shape",
                "1,32,32",
                "--overwrite",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Check that zarr store was created
        zarr_files = list(setup["output_dir"].glob("*.zarr"))
        assert len(zarr_files) == 1

        zarr_path = zarr_files[0]
        assert zarr_path.is_dir()

    def test_convert_command_with_options(self, cli_test_setup):
        """Test convert command with various options."""
        setup = cli_test_setup
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                "convert",
                str(setup["images_dir"]),
                "--metadata",
                str(setup["metadata_path"]),
                "--out",
                str(setup["output_dir"]),
                "--recursive",
                "--workers",
                "1",
                "--chunk-shape",
                "1,16,16",
                "--compressor",
                "lz4",
                "--clevel",
                "1",
                "--overwrite",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

    def test_convert_command_without_metadata(self, cli_test_setup):
        """Test convert CLI command without providing metadata."""
        setup = cli_test_setup
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                "convert",
                str(setup["images_dir"]),
                "--out",
                str(setup["output_dir"]),
                "--workers",
                "2",
                "--overwrite",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Check that the store was created with default name
        zarr_store = setup["output_dir"] / "images.zarr"
        assert zarr_store.exists()

    def test_inspect_command(self, cli_test_setup):
        """Test the inspect CLI command."""
        setup = cli_test_setup
        runner = CliRunner()

        # First create a store
        result = runner.invoke(
            main,
            [
                "convert",
                str(setup["images_dir"]),
                "--metadata",
                str(setup["metadata_path"]),
                "--out",
                str(setup["output_dir"]),
                "--overwrite",
            ],
        )
        assert result.exit_code == 0

        # Find the created zarr store
        zarr_files = list(setup["output_dir"].glob("*.zarr"))
        assert len(zarr_files) == 1
        zarr_path = zarr_files[0]

        # Then inspect it
        result = runner.invoke(main, ["inspect", str(zarr_path)])

        assert result.exit_code == 0, f"Inspect failed: {result.output}"
        assert "SUMMARY STATISTICS" in result.output
        assert f"Total images across all files: {len(setup['files'])}" in result.output

    def test_cli_help(self):
        """Test CLI help commands."""
        runner = CliRunner()

        # Test main help
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "convert" in result.output
        assert "inspect" in result.output

        # Test convert help
        result = runner.invoke(main, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--metadata" in result.output
        assert "--out" in result.output

        # Test inspect help
        result = runner.invoke(main, ["inspect", "--help"])
        assert result.exit_code == 0

    def test_cli_error_handling(self, cli_test_setup):
        """Test CLI error handling."""
        setup = cli_test_setup
        runner = CliRunner()

        # Test with missing metadata file
        result = runner.invoke(
            main,
            [
                "convert",
                str(setup["images_dir"]),
                "--metadata",
                str(setup["temp_dir"] / "nonexistent.csv"),
                "--out",
                str(setup["output_dir"]),
            ],
        )

        assert result.exit_code != 0

        # Test with non-existent input directory
        result = runner.invoke(
            main,
            [
                "convert",
                str(setup["temp_dir"] / "nonexistent"),
                "--metadata",
                str(setup["metadata_path"]),
                "--out",
                str(setup["output_dir"]),
            ],
        )

        assert result.exit_code != 0

    def test_version_option(self):
        """Test the version option."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        # Should contain version info
