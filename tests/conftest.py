import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
from images_to_zarr import configure_logging

configure_logging(enable=True, level="WARNING")  # Reduce noise in tests
