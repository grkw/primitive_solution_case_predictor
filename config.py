from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:

    # After trimming
    csv_input_col: int = 7 # upper bound is exclusive
    csv_label_col: str = "ruckig_solution"

@dataclass
class TrainConfig(Config):
    train_csv_file: List[str] = field(default_factory=lambda: ['data/test_10scale_6seed_5vmax_5amax_15jmax.csv'])
    val_csv_file: List[str] = field(default_factory=lambda: ['data/val_data_8192paths_8wps_1scale_EXPplanner_2seed.csv'])
    
    num_examples_per_class: int = 100
    batch_size: int = 64
    input_size: int = 18
    output_size: int = 32

    num_epochs: int = 500

@dataclass
class TestConfig(Config):    
    test_csv_file: List[str] = field(default_factory=lambda: ['data/test_data_16384paths_8wps_1scale_EXPplanner_3seed.csv'])
    test_model_path: str = ''

    batch_size: int = 64
    
    num_examples_per_class: int = 50
    
    # af_desc_col: int = 19