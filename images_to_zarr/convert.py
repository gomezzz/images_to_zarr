from __future__ import annotations
from pathlib import Path
from typing import Sequence


def convert(
    folders: Sequence[Path] | Sequence[str],
    recursive: bool,
    metadata: Path | str,
    output_dir: Path | str,
    num_parallel_workers: int = 8,
    fits_extension: int | str | Sequence[int | str] | None = None,
    *,
    chunk_shape: tuple[int, int, int] = (1, 256, 256),
    shard_bytes: int = 16 * 2**20,
    compressor: str = "zstd",
    clevel: int = 4,
    overwrite: bool = False,
) -> Path:
    """
    Re-package a heterogeneous image collection (FITS/PNG/JPEG/TIFF) plus
    tabular metadata into a *single* **sharded Zarr v3** store.

    Parameters
    ----------
    folders
        One or more directories containing images.
    recursive
        If *True*, scan sub-directories too.
    metadata
        CSV file with at least a ``filename`` column; additional fields
        (e.g. ``source_id``, ``ra``, ``dec`` …) are copied verbatim into
        a Parquet side-car and attached as Zarr attributes for easy joins.
    output_dir
        Destination path; a directory called ``<name>.zarr`` is created
        inside it.  Existing stores are refused unless *overwrite* is set.
    num_parallel_workers
        Threads or processes used to ingest images and write chunks.
    fits_extension
        Which FITS HDU(s) to read:

        * ``None``  →  use extension 0
        * *int* or *str*  →  single HDU
        * *Sequence*  →  concatenate multiple HDUs along the channel axis
    chunk_shape
        Chunk layout **(n_images, height, width)** ; the first dimension
        **must be 1** so each image maps to exactly one chunk.
    shard_bytes
        Target size (bytes) of each shard container file.
    compressor
        Any *numcodecs* codec name (``"zstd"``, ``"lz4"``, …).
    clevel
        Compression level handed to *numcodecs*.
    overwrite
        Destroy an existing store at *output_dir* if present.

    Returns
    -------
    Path
        Path to the root of the new ``*.zarr`` store.

    Notes
    -----
    * The function is purely I/O bound; if the host has a fast network
      file-system prefer a *ThreadPoolExecutor*.
    * A sibling file ``metadata.parquet`` is always written – fast joins,
      Arrow-native.
    * Sharding keeps the inode count roughly equal to “1 000 HDF5 files”
      for 100 M images but remains S3-friendly.
    """
    pass
