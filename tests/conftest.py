import sys
from pathlib import Path
from images_to_zarr import configure_logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
configure_logging(enable=True, level="WARNING")  # Reduce noise in tests
