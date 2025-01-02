from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProjectConfig:
    # Root paths
    PROJECT_ROOT: Path = '/home/marijatochadse/1_data'

    DATA_ROOT: Path = PROJECT_ROOT / "data"
    RAW_DATA: Path = DATA_ROOT / "raw"
    PROCESSED_DATA: Path = DATA_ROOT / "processed"
    MODELS_PATH: Path = PROJECT_ROOT / "models"

    # Model parameters
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2



configs = ProjectConfig()