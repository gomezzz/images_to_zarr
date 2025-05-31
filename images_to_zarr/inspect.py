from pathlib import Path


def inspect(store: Path | str) -> None:
    """
    Print a human-readable summary of a sharded Zarr image archive.

    The layout and wording mimic the example you supplied.

    Parameters
    ----------
    store
        Path pointing to the ``*.zarr`` directory.

    Examples
    --------
    >>> inspect_zarr(\"~/data/galaxy_cutouts.zarr\")
    ================================================================================
    SUMMARY STATISTICS
    ================================================================================
    Total images across all files: 104 857 600
    Total storage size: 126 743.31 MB
    Average file size: 126.74 MB
    File size range:  8.12 – 531.00 MB

    Format distribution:
      FITS:  60 000 000 (57.2 %)
      PNG:   30 000 000 (28.6 %)
      JPEG:  10 000 000 ( 9.5 %)
      TIFF:   4 857 600 ( 4.6 %)

    Original data type distribution:
      uint8:   78 %
      int16:   12 %
      float32: 10 %
    """
    pass
