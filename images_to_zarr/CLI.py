import click


from .convert import convert as _convert
from .inspect import inspect as inspect_zarr


@click.group()
@click.version_option()
def main() -> None:
    pass


@main.command()
@click.argument("folders", type=click.Path(exists=True), nargs=-1, required=False)
@click.option(
    "--out", "output_dir", type=click.Path(), required=True, help="Output directory for Zarr store"
)
@click.option("--recursive", is_flag=True, help="Scan subdirectories recursively")
@click.option(
    "--metadata",
    type=click.Path(exists=True),
    required=False,
    help="Optional CSV file with metadata including 'filename' column. "
    "If not provided, metadata will be created from filenames only.",
)
@click.option("--workers", "num_parallel_workers", default=8, help="Number of parallel workers")
@click.option(
    "--fits-ext", "fits_extension", default=None, help="FITS extension to read (number or name)"
)
@click.option("--chunk-shape", default="1,256,256", help="Chunk shape as comma-separated values")
@click.option("--compressor", default="lz4", help="Compression codec")
@click.option("--clevel", default=1, help="Compression level")
@click.option("--overwrite", is_flag=True, help="Overwrite existing store")
@click.option(
    "--resize",
    default=None,
    help="Resize all images to specified dimensions as 'height,width' (e.g., '512,512'). "
    "If not specified, all images must have the same dimensions.",
)
@click.option(
    "--interpolation-order",
    default=1,
    type=click.IntRange(0, 5),
    help="Interpolation order for resizing: 0=nearest, 1=linear, 2=quadratic, 3=cubic, 4=quartic, 5=quintic",
)
def convert(**kw):
    """Convert image folders to Zarr format."""
    # Parse chunk shape
    chunk_shape_str = kw.pop("chunk_shape", "1,256,256")
    chunk_shape = tuple(map(int, chunk_shape_str.split(",")))

    # Handle fits extension
    fits_ext = kw.get("fits_extension")
    if fits_ext is not None:
        try:
            fits_ext = int(fits_ext)
        except ValueError:
            pass  # Keep as string
        kw["fits_extension"] = fits_ext  # Handle resize parameter
    resize_str = kw.pop("resize", None)
    if resize_str is not None:
        try:
            height, width = map(int, resize_str.split(","))
            kw["resize"] = (height, width)
        except ValueError:
            raise click.ClickException(
                f"Invalid resize format '{resize_str}'. Use 'height,width' (e.g., '512,512')"
            )
    else:
        kw["resize"] = None

    kw["chunk_shape"] = chunk_shape

    # Convert folders to list if provided
    folders = kw.pop("folders", ())
    if folders:
        kw["folders"] = list(folders)
    else:
        kw["folders"] = None

    # Call convert with proper parameter order
    result = _convert(**kw)
    if result:
        click.echo(f"Successfully created Zarr store: {result}")
    else:
        click.echo("Conversion completed")


@main.command()
@click.argument("store", type=click.Path(exists=True))
def inspect(store):
    """Inspect a Zarr store and print statistics."""
    inspect_zarr(store)
