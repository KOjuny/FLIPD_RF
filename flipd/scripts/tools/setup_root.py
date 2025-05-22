import os
import sys
from pathlib import Path


def setup_root():
    # Add the root directory to path (no matter where the script is run from)
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    sys.path.append(".")
