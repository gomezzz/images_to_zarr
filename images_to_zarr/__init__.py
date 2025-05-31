# Supported file extensions for images to Zarr conversion
I2Z_SUPPORTED_EXTS = {".fits", ".fit", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Version
__version__ = "0.1.0"

# Logging configuration
import sys
from loguru import logger


# Configure logger with option to disable for releases
def configure_logging(enable: bool = True, level: str = "INFO"):
    """Configure loguru logging.

    Parameters
    ----------
    enable : bool
        If False, disable all logging output
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    if not enable:
        logger.remove()
        logger.add(sys.stderr, level="CRITICAL")  # Only critical errors
    else:
        logger.remove()
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
                "<cyan>{line}</cyan> - <level>{message}</level>"
            ),
            level=level,
            colorize=True,
        )


# Initialize with default settings
configure_logging(enable=True)

from .convert import convert
from .inspect import inspect

all = [
    "convert",
    "inspect",
    "__version__",
]
