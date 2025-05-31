import click


from .convert import convert as _convert
from .inspect import inspect as inspect_zarr


@click.group()
@click.version_option()
def main() -> None:
    pass


@main.command()
@click.argument("folders", type=click.Path(exists=True), nargs=-1, required=True)
@click.option("--recursive", is_flag=True)
@click.option("--metadata", type=click.Path(exists=True), required=True)
@click.option("--out", "output_dir", type=click.Path(), required=True)
@click.option("--workers", default=8)
@click.option("--fits-ext", "fits_extension", default=None)
def convert(**kw):
    _convert(**kw)


@main.command()
@click.argument("store", type=click.Path(exists=True))
def inspect(store):
    inspect_zarr(store)
