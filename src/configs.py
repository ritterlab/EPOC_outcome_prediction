from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProjectConfig:
    # Root paths
    PROJECT_ROOT: Path = '/home/marijatochadse/1_data'
    OUTPUT_ROOT: Path = '/home/marijatochadse/outputs'


    # Model parameters
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2



configs = ProjectConfig()