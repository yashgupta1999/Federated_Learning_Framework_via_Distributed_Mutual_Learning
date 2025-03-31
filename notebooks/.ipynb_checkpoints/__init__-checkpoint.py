import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import commonly used modules
from data_utils import *
from model_utils import *
from config_utils import load_config

# Load default config
CONFIG = load_config('../config/fl_template_config.yaml')

# Export commonly used items
__all__ = ['CONFIG']